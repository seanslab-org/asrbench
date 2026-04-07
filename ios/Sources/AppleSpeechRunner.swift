import Foundation
import Speech
import AVFoundation

/// Wraps SFSpeechRecognizer with `requiresOnDeviceRecognition = true`.
///
/// Fails loudly if on-device recognition is not available for the requested
/// locale on the current Xcode/simulator combo — we never want to silently fall
/// back to server mode (audio uploaded to Apple) for benchmarking.
final class AppleSpeechRunner {
    let runnerName: String
    let locale: Locale
    private let recognizer: SFSpeechRecognizer

    enum AppleSpeechError: Error, CustomStringConvertible {
        case recognizerInitFailed(String)
        case onDeviceUnavailable(String)
        case authorizationDenied(SFSpeechRecognizerAuthorizationStatus)
        case recognitionFailed(String)
        case unknownLocale(String)

        var description: String {
            switch self {
            case .recognizerInitFailed(let msg): return "recognizerInitFailed: \(msg)"
            case .onDeviceUnavailable(let loc): return "onDeviceUnavailable for locale \(loc) — will not silently use server mode"
            case .authorizationDenied(let status): return "authorizationDenied (\(status.rawValue))"
            case .recognitionFailed(let msg): return "recognitionFailed: \(msg)"
            case .unknownLocale(let loc): return "unknownLocale: \(loc)"
            }
        }
    }

    init(localeIdentifier: String = "en-US") throws {
        self.locale = Locale(identifier: localeIdentifier)
        self.runnerName = "apple-speech-\(localeIdentifier)"
        guard let r = SFSpeechRecognizer(locale: locale) else {
            throw AppleSpeechError.unknownLocale(localeIdentifier)
        }
        self.recognizer = r

        // Hard requirement: on-device only.
        if !r.supportsOnDeviceRecognition {
            throw AppleSpeechError.onDeviceUnavailable(localeIdentifier)
        }
    }

    /// Call once on app launch. Blocks the calling task until auth resolves.
    static func requestAuthorization() async throws {
        let status: SFSpeechRecognizerAuthorizationStatus = await withCheckedContinuation { cont in
            SFSpeechRecognizer.requestAuthorization { cont.resume(returning: $0) }
        }
        switch status {
        case .authorized:
            return
        default:
            throw AppleSpeechError.authorizationDenied(status)
        }
    }

    /// Transcribe a file URL synchronously (returns final result text and confidence).
    /// Confidence is the average per-segment confidence reported by Apple.
    func transcribe(audioURL: URL) async throws -> (String, Double?) {
        let request = SFSpeechURLRecognitionRequest(url: audioURL)
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = false
        // Disable automatic punctuation if available — we want raw words for fair WER
        if #available(iOS 16.0, *) {
            request.addsPunctuation = false
        }

        return try await withCheckedThrowingContinuation { cont in
            recognizer.recognitionTask(with: request) { result, error in
                if let error {
                    cont.resume(throwing: AppleSpeechError.recognitionFailed("\(error)"))
                    return
                }
                guard let result, result.isFinal else { return }
                let best = result.bestTranscription
                let avgConfidence: Double? = {
                    let confs = best.segments.map { Double($0.confidence) }.filter { $0 > 0 }
                    return confs.isEmpty ? nil : confs.reduce(0, +) / Double(confs.count)
                }()
                cont.resume(returning: (best.formattedString, avgConfidence))
            }
        }
    }
}
