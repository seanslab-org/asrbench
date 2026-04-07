import Foundation
import AVFoundation
import MoonshineVoice

/// Orchestrates the benchmark: loads the manifest, runs each registered runner
/// over each sample, writes results JSON to the app's Documents directory.
@MainActor
final class BenchRunner: ObservableObject {

    enum Phase: Equatable {
        case idle
        case authorizing
        case loading(String)
        case running(runner: String, current: Int, total: Int)
        case writing
        case done(path: String, summary: String)
        case failed(String)
    }

    @Published var phase: Phase = .idle

    func run() async {
        do {
            phase = .authorizing
            try await AppleSpeechRunner.requestAuthorization()

            phase = .loading("manifest.json")
            let manifest = try loadManifest()
            print("Loaded \(manifest.count) samples")

            // Build runners. Apple Speech first because it's lightest to fail-fast.
            var runners: [String: (URL) async throws -> (String, Double?)] = [:]
            phase = .loading("apple-speech")
            let apple = try AppleSpeechRunner(localeIdentifier: "en-US")
            runners[apple.runnerName] = { url in
                try await apple.transcribe(audioURL: url)
            }

            phase = .loading("moonshine-tiny-en")
            let mTiny = try MoonshineRunner(arch: "tiny-en", modelArch: .tiny)
            runners[mTiny.runnerName] = { url in
                let t = try mTiny.transcribe(audioURL: url)
                return (t, nil)
            }

            phase = .loading("moonshine-base-en")
            let mBase = try MoonshineRunner(arch: "base-en", modelArch: .base)
            runners[mBase.runnerName] = { url in
                let t = try mBase.transcribe(audioURL: url)
                return (t, nil)
            }

            // Run sequentially: per-runner outer loop so each runner has a hot
            // model and doesn't fight CUDA-equivalent caches with the others.
            var results: [BenchResult] = []
            let runnerOrder = ["apple-speech-en-US", mTiny.runnerName, mBase.runnerName]
            for runnerName in runnerOrder {
                guard let fn = runners[runnerName] else { continue }
                for (i, entry) in manifest.enumerated() {
                    phase = .running(runner: runnerName, current: i + 1, total: manifest.count)
                    let url = audioURL(for: entry.filename)
                    let dur = audioDuration(url: url)

                    let t0 = Date()
                    do {
                        let (text, conf) = try await fn(url)
                        let elapsed = Date().timeIntervalSince(t0)
                        results.append(BenchResult(
                            runner: runnerName,
                            sampleId: entry.sampleId,
                            language: entry.language,
                            audioDurationS: dur,
                            latencyS: elapsed,
                            transcript: text,
                            confidence: conf,
                            error: nil
                        ))
                    } catch {
                        let elapsed = Date().timeIntervalSince(t0)
                        results.append(BenchResult(
                            runner: runnerName,
                            sampleId: entry.sampleId,
                            language: entry.language,
                            audioDurationS: dur,
                            latencyS: elapsed,
                            transcript: "",
                            confidence: nil,
                            error: "\(error)"
                        ))
                    }
                }
            }

            phase = .writing
            let path = try writeResults(results)
            let summary = makeSummary(results)
            phase = .done(path: path, summary: summary)
            print("Wrote results to \(path)")
            print(summary)
        } catch {
            phase = .failed("\(error)")
            print("BenchRunner failed: \(error)")
        }
    }

    // MARK: - Helpers

    private func loadManifest() throws -> [ManifestEntry] {
        guard let url = Bundle.main.url(forResource: "audio/manifest", withExtension: "json") else {
            throw NSError(domain: "BenchRunner", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "manifest.json not found in bundle"])
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([ManifestEntry].self, from: data)
    }

    private func audioURL(for filename: String) -> URL {
        return Bundle.main.bundleURL
            .appendingPathComponent("audio")
            .appendingPathComponent(filename)
    }

    private func audioDuration(url: URL) -> Double {
        guard let file = try? AVAudioFile(forReading: url) else { return 0 }
        let frames = Double(file.length)
        let sr = file.fileFormat.sampleRate
        return sr > 0 ? frames / sr : 0
    }

    private func writeResults(_ results: [BenchResult]) throws -> String {
        let formatter = ISO8601DateFormatter()
        let stamp = formatter.string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let outURL = docs.appendingPathComponent("results-\(stamp).json")

        let device = "\(ProcessInfo.processInfo.operatingSystemVersionString)"
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?"

        let output = BenchOutput(
            timestamp: ISO8601DateFormatter().string(from: Date()),
            device: device,
            xcodeBuild: build,
            results: results
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(output)
        try data.write(to: outURL, options: .atomic)
        return outURL.path
    }

    private func makeSummary(_ results: [BenchResult]) -> String {
        var byRunner: [String: [BenchResult]] = [:]
        for r in results { byRunner[r.runner, default: []].append(r) }
        var lines: [String] = []
        lines.append("=== bench summary (host-side WER computation needed) ===")
        for (runner, rs) in byRunner.sorted(by: { $0.key < $1.key }) {
            let n = rs.count
            let errs = rs.filter { $0.error != nil }.count
            let validRTF = rs.filter { $0.error == nil }
            let avgRTF = validRTF.isEmpty ? 0
                : validRTF.map { $0.rtf }.reduce(0, +) / Double(validRTF.count)
            lines.append("  \(runner): n=\(n) errors=\(errs) avgRTF=\(String(format: "%.3f", avgRTF))")
        }
        return lines.joined(separator: "\n")
    }
}
