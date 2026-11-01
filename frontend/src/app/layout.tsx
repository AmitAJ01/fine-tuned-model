import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FineTuneOps | DevOps Incident Diagnosis Platform",
  description: "End-to-End LLM Fine-Tuning & Evaluation Platform for DevOps Incident Diagnosis",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased min-h-screen bg-slate-950 text-slate-100">
        <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <span className="h-8 w-8 rounded-lg bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center font-mono font-bold text-emerald-400">
                FT
              </span>
              <span className="font-semibold text-lg tracking-tight">FineTuneOps</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700">
                v0.1.0-alpha
              </span>
            </div>
            <nav className="flex items-center space-x-6 text-sm text-slate-300">
              <a href="#diagnose" className="hover:text-emerald-400 transition">Diagnosis</a>
              <a href="#benchmark" className="hover:text-emerald-400 transition">Baseline vs Fine-Tuned</a>
              <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" className="hover:text-emerald-400 transition">
                API Docs
              </a>
            </nav>
          </div>
        </header>
        <main className="max-w-7xl mx-auto px-4 py-8">{children}</main>
      </body>
    </html>
  );
}
