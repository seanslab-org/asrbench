import Foundation
import MoonshineVoice

/// Wraps `MoonshineVoice.Transcriber` for our benchmark loop.
///
/// The model directory is expected at
/// `Bundle.main.url(forResource: "moonshine-models/<arch>", withExtension: nil)`.
/// xcodegen `project.yml` adds `Resources/moonshine-models` as a folder reference.
final class MoonshineRunner {
    let runnerName: String
    private let transcriber: Transcriber

    enum MoonshineRunnerError: Error, CustomStringConvertible {
        case modelDirNotFound(String)
        case loadFailed(String)
        case transcribeFailed(String)

        var description: String {
            switch self {
            case .modelDirNotFound(let p): return "modelDirNotFound: \(p)"
            case .loadFailed(let m): return "loadFailed: \(m)"
            case .transcribeFailed(let m): return "transcribeFailed: \(m)"
            }
        }
    }

    /// `arch` is one of `tiny-en`, `base-en`, etc. — must match the directory name
    /// inside `Resources/moonshine-models/`.
    init(arch: String, modelArch: ModelArch) throws {
        self.runnerName = "moonshine-\(arch)"
        guard let modelURL = Bundle.main.url(
            forResource: "moonshine-models/\(arch)",
            withExtension: nil
        ) else {
            throw MoonshineRunnerError.modelDirNotFound(arch)
        }
        do {
            self.transcriber = try Transcriber(
                modelPath: modelURL.path,
                modelArch: modelArch
            )
        } catch {
            throw MoonshineRunnerError.loadFailed("\(error)")
        }
    }

    /// Transcribe a WAV file. Returns the concatenated text of all transcript lines.
    /// Note: Moonshine's WAVLoader supports 16/24/32-bit PCM mono/stereo.
    func transcribe(audioURL: URL) throws -> String {
        let wav: WAVData
        do {
            wav = try loadWAVFile(audioURL.path)
        } catch {
            throw MoonshineRunnerError.transcribeFailed("wav load: \(error)")
        }

        let transcript: Transcript
        do {
            transcript = try transcriber.transcribeWithoutStreaming(
                audioData: wav.audioData,
                sampleRate: Int32(wav.sampleRate)
            )
        } catch {
            throw MoonshineRunnerError.transcribeFailed("transcribe: \(error)")
        }

        // Concatenate all lines into one string. Moonshine generally produces one
        // line for short utterances; multi-line is for long form / streaming.
        return transcript.lines.map { $0.text }.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    deinit {
        transcriber.close()
    }
}
