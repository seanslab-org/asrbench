import SwiftUI

@main
struct AsrBenchApp: App {
    @StateObject private var runner = BenchRunner()

    var body: some Scene {
        WindowGroup {
            ContentView(runner: runner)
                .task {
                    // Auto-run on launch — this is a benchmark app, not interactive
                    await runner.run()
                }
        }
    }
}

struct ContentView: View {
    @ObservedObject var runner: BenchRunner

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("AsrBench iOS")
                .font(.title)
                .padding(.top)
            statusView
            Spacer()
            Text("Pull results via:\nxcrun simctl get_app_container booted ai.moonshine.asrbench data")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding()
    }

    @ViewBuilder
    private var statusView: some View {
        switch runner.phase {
        case .idle:
            Text("idle")
        case .authorizing:
            ProgressView("Requesting Speech authorization…")
        case .loading(let what):
            ProgressView("Loading \(what)…")
        case .running(let r, let cur, let total):
            VStack(alignment: .leading) {
                Text("Running \(r)")
                ProgressView(value: Double(cur), total: Double(total))
                Text("\(cur)/\(total)").font(.caption)
            }
        case .writing:
            ProgressView("Writing results…")
        case .done(let path, let summary):
            VStack(alignment: .leading, spacing: 4) {
                Text("Done").font(.headline).foregroundStyle(.green)
                Text(path).font(.caption2).lineLimit(2)
                Text(summary).font(.system(.caption, design: .monospaced))
            }
        case .failed(let msg):
            VStack(alignment: .leading, spacing: 4) {
                Text("Failed").font(.headline).foregroundStyle(.red)
                Text(msg).font(.caption).lineLimit(nil)
            }
        }
    }
}
