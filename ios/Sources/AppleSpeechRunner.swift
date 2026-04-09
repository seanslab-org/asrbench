import Foundation
import Speech
import AVFoundation

/// Apple's on-device speech recognition runner.
///
/// Strategy by platform:
/// - **Real device (iPad/iPhone):** Uses the legacy `SFSpeechRecognizer` with
///   `requiresOnDeviceRecognition = true`. This works on real devices where the
///   speech model is pre-installed. The newer `SpeechTranscriber` API (iOS 26)
///   crashes with EXC_BREAKPOINT on iPad mini 6 (A15) — Apple hasn't added it
///   to their hardware allowlist yet.
/// - **Simulator:** Neither API works. The simulator doesn't ship the speech
///   model assets (`kLSRErrorDomain Code=300` for legacy, `Status.unsupported`
///   for new API). See `results/ios_simulator_20260408/README.md`.
final class AppleSpeechRunner {
    let runnerName: String
    let locale: Locale
    private let recognizer: SFSpeechRecognizer

    enum AppleSpeechError: Error, CustomStringConvertible {
        case recognizerInitFailed(String)
        case onDeviceUnavailable(String)
        case authorizationDenied(SFSpeechRecognizerAuthorizationStatus)
        case recognitionFailed(String)

        var description: String {
            switch self {
            case .recognizerInitFailed(let msg): return "recognizerInitFailed: \(msg)"
            case .onDeviceUnavailable(let loc): return "onDeviceUnavailable: \(loc)"
            case .authorizationDenied(let s): return "authorizationDenied (\(s.rawValue))"
            case .recognitionFailed(let msg): return "recognitionFailed: \(msg)"
            }
        }
    }

    init(localeIdentifier: String = "en-US") throws {
        self.locale = Locale(identifier: localeIdentifier)
        self.runnerName = "apple-speech-\(localeIdentifier)"
        guard let r = SFSpeechRecognizer(locale: locale) else {
            throw AppleSpeechError.recognizerInitFailed(localeIdentifier)
        }
        self.recognizer = r
    }

    /// Request Speech permission. Must be called before transcribe().
    /// On real device this shows a system dialog (or uses TCC pre-grant).
    static func requestAuthorization() async throws {
        let status: SFSpeechRecognizerAuthorizationStatus = await withCheckedContinuation { cont in
            SFSpeechRecognizer.requestAuthorization { cont.resume(returning: $0) }
        }
        guard status == .authorized else {
            throw AppleSpeechError.authorizationDenied(status)
        }
    }

    /// Check if on-device recognition is available. On simulator this
    /// returns true but the actual recognition still fails with Code=300.
    /// On real devices it genuinely indicates model availability.
    var supportsOnDevice: Bool {
        recognizer.supportsOnDeviceRecognition
    }

    /// Transcribe a file URL. Prefers on-device but allows server fallback
    /// so we at least get Apple's accuracy baseline even if the on-device
    /// model hasn't been downloaded yet.
    func transcribe(audioURL: URL) async throws -> (String, Double?) {
        let request = SFSpeechURLRecognitionRequest(url: audioURL)
        // Try on-device first; if model not present, allow server fallback
        request.requiresOnDeviceRecognition = recognizer.supportsOnDeviceRecognition
        request.shouldReportPartialResults = false
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
