import Foundation

/// One per-sample, per-runner result row. Designed to be JSON-encoded
/// in the iOS app and consumed by `scripts/compute_metrics.py` on the host.
public struct BenchResult: Codable {
    public let runner: String           // "apple-speech-en", "moonshine-tiny-en", "moonshine-base-en"
    public let sampleId: String         // matches manifest.json sample_id
    public let language: String         // "en"
    public let audioDurationS: Double   // wall-clock audio length
    public let latencyS: Double         // transcribe wall-clock (load excluded)
    public let rtf: Double              // latencyS / audioDurationS
    public let transcript: String       // raw model output (un-normalized)
    public let confidence: Double?      // present for Apple Speech, nil for Moonshine
    public let error: String?           // non-nil if the runner threw

    public init(
        runner: String,
        sampleId: String,
        language: String,
        audioDurationS: Double,
        latencyS: Double,
        transcript: String,
        confidence: Double?,
        error: String?
    ) {
        self.runner = runner
        self.sampleId = sampleId
        self.language = language
        self.audioDurationS = audioDurationS
        self.latencyS = latencyS
        self.rtf = audioDurationS > 0 ? latencyS / audioDurationS : 0
        self.transcript = transcript
        self.confidence = confidence
        self.error = error
    }
}

/// JSON the iOS app reads from `Resources/audio/manifest.json`.
public struct ManifestEntry: Codable {
    public let sampleId: String
    public let filename: String         // relative to Resources/audio/
    public let reference: String
    public let language: String
}

public struct BenchOutput: Codable {
    public let timestamp: String
    public let device: String           // UIDevice.current.name + iOS version
    public let xcodeBuild: String       // bundle build/version
    public let results: [BenchResult]
}
