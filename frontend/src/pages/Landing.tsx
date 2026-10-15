import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion, useReducedMotion } from "motion/react";

import { TextFlippingBoard } from "@/components/ui/text-flipping-board";

const BOARD_MESSAGES = [
  "POWER PREDICT",
  "NEXT-HOUR\nELECTRICITY\nFOR BUILDINGS",
  "EXPLAINED,\nNOT GUESSED.",
] as const;

const PIPELINE = [
  {
    step: "01",
    title: "Ingest",
    body: "Load BDG2 cleaned electricity for three education buildings, plus matching weather.",
  },
  {
    step: "02",
    title: "Features",
    body: "Lags, rolling windows, calendar, and weather — chronological only, never shuffled.",
  },
  {
    step: "03",
    title: "Train",
    body: "Offline CLI fits Linear Regression, Random Forest, XGBoost, LSTM, and CNN-LSTM.",
  },
  {
    step: "04",
    title: "Evaluate",
    body: "MAE, RMSE, R², and MAPE on a held-out test window. Metrics come from real runs.",
  },
  {
    step: "05",
    title: "Explain",
    body: "SHAP for why a prediction landed where it did. Occupancy stays labeled simulated.",
  },
  {
    step: "06",
    title: "Serve",
    body: "FastAPI + React desk: forecasts, what-if, anomalies, and an IoT simulator path.",
  },
] as const;

const MODELS = [
  { name: "Linear Regression", kind: "Baseline" },
  { name: "Random Forest", kind: "Baseline" },
  { name: "XGBoost", kind: "Baseline" },
  { name: "LSTM", kind: "Deep learning" },
  { name: "CNN-LSTM", kind: "Deep learning" },
] as const;

const fadeUp = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0 },
};

export function LandingPage() {
  const reduceMotion = useReducedMotion();
  const [msgIdx, setMsgIdx] = useState(0);
  const [pipelineProgress, setPipelineProgress] = useState(0);

  const nextBoard = useCallback(() => {
    setMsgIdx((i) => (i + 1) % BOARD_MESSAGES.length);
  }, []);

  useEffect(() => {
    if (reduceMotion) return;
    const id = window.setInterval(nextBoard, 5200);
    return () => window.clearInterval(id);
  }, [nextBoard, reduceMotion]);

  useEffect(() => {
    if (reduceMotion) {
      setPipelineProgress(100);
      return;
    }
    setPipelineProgress(0);
    const id = window.setInterval(() => {
      setPipelineProgress((p) => (p >= 100 ? 0 : p + 2));
    }, 90);
    return () => window.clearInterval(id);
  }, [reduceMotion]);

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-black text-zinc-100">
      <div
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top,_rgba(212,137,74,0.12),_transparent_55%)]"
        aria-hidden
      />
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.07]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px)",
          backgroundSize: "48px 48px",
        }}
        aria-hidden
      />

      <header className="relative z-10 mx-auto flex max-w-6xl items-center justify-between gap-4 px-6 pt-8 sm:px-8">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-copper">
            Academic project
          </p>
          <p className="font-display text-xl font-extrabold tracking-tight text-white">
            PowerPredict
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Link
            to="/login"
            className="border border-zinc-700 px-3 py-2 font-display text-xs uppercase tracking-[0.14em] text-zinc-100 hover:border-copper hover:text-white"
          >
            Sign in
          </Link>
          <Link
            to="/login"
            className="bg-copper px-3 py-2 font-display text-xs font-bold uppercase tracking-[0.14em] text-black hover:brightness-110"
          >
            Open desk
          </Link>
        </div>
      </header>

      <main className="relative z-10 mx-auto max-w-6xl px-6 pb-24 pt-14 sm:px-8 sm:pt-20">
        {/* Hero */}
        <section className="flex flex-col items-center text-center">
          <motion.p
            className="font-mono text-[11px] uppercase tracking-[0.24em] text-zinc-400"
            initial={reduceMotion ? false : { opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            Smart-building energy intelligence
          </motion.p>

          <motion.div
            className="mt-8 w-full"
            initial={reduceMotion ? false : { opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            <TextFlippingBoard
              text={BOARD_MESSAGES[msgIdx]}
              className="border border-zinc-800 !bg-zinc-950 shadow-[0_24px_80px_-20px_rgba(0,0,0,0.9)]"
              duration={1.4}
            />
          </motion.div>

          <motion.p
            className="mt-8 max-w-2xl text-base leading-7 text-zinc-300 sm:text-lg"
            initial={reduceMotion ? false : { opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.25 }}
          >
            Next-hour electricity for buildings — explained, not guessed.
          </motion.p>
          <p className="mt-3 max-w-xl text-sm leading-6 text-zinc-400">
            Offline-trained models on Building Data Genome Project 2. The desk loads saved
            artifacts for forecasts, what-if runs, SHAP, and anomalies. Occupancy and MQTT IoT
            paths are labeled <span className="text-sim">simulated</span> — not live meters.
          </p>

          <div className="mt-10 flex flex-wrap items-center justify-center gap-3">
            <Link
              to="/login"
              className="bg-copper px-6 py-3 font-display text-sm font-bold text-black hover:brightness-110"
            >
              Sign in to open desk
            </Link>
            <a
              href="#pipeline"
              className="border border-zinc-700 px-6 py-3 font-display text-sm text-zinc-100 hover:border-copper"
            >
              See the pipeline
            </a>
          </div>
        </section>

        {/* Pipeline */}
        <section id="pipeline" className="mt-24 scroll-mt-20">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-copper">
                How it runs
              </p>
              <h2 className="mt-2 font-display text-2xl font-bold tracking-tight text-white sm:text-3xl">
                From dataset to desk
              </h2>
            </div>
            <p className="max-w-sm text-sm leading-6 text-zinc-400">
              Training is offline. The UI never invents scores — empty tiles mean missing
              artifacts.
            </p>
          </div>

          <div className="mt-6 h-1.5 overflow-hidden rounded-full bg-zinc-900">
            <motion.div
              className="h-full rounded-full bg-gradient-to-r from-copper via-amber-300 to-phosphor"
              style={{ width: `${pipelineProgress}%` }}
              aria-hidden
            />
          </div>
          <p className="mt-2 font-mono text-[10px] uppercase tracking-[0.18em] text-zinc-500">
            Pipeline progress · demo loop
          </p>

          <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {PIPELINE.map((item, i) => {
              const active = pipelineProgress >= ((i + 1) / PIPELINE.length) * 100 - 2;
              return (
                <motion.article
                  key={item.step}
                  className={`border px-4 py-4 transition-colors ${
                    active
                      ? "border-copper/60 bg-zinc-950"
                      : "border-zinc-800 bg-black/60"
                  }`}
                  variants={fadeUp}
                  initial={reduceMotion ? false : "hidden"}
                  whileInView="show"
                  viewport={{ once: true, margin: "-40px" }}
                  transition={{ duration: 0.4, delay: i * 0.05 }}
                >
                  <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-copper">
                    {item.step}
                  </p>
                  <h3 className="mt-2 font-display text-lg font-semibold text-white">
                    {item.title}
                  </h3>
                  <p className="mt-2 text-sm leading-6 text-zinc-400">{item.body}</p>
                </motion.article>
              );
            })}
          </div>
        </section>

        {/* Dataset */}
        <section className="mt-24 grid gap-10 border-t border-zinc-800 pt-16 lg:grid-cols-[1.1fr_0.9fr]">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-copper">
              Dataset
            </p>
            <h2 className="mt-2 font-display text-2xl font-bold tracking-tight text-white sm:text-3xl">
              Building Data Genome Project 2
            </h2>
            <p className="mt-4 max-w-xl text-sm leading-7 text-zinc-300">
              Public hourly electricity meters (2016–2017). This demo uses three{" "}
              <span className="text-white">Hog</span> site education buildings with high
              completeness — historical public meters, not live GPS weather for your location.
            </p>
            <ul className="mt-6 space-y-2 font-mono text-sm text-zinc-200">
              <li className="flex gap-3 border-b border-zinc-800 py-2">
                <span className="text-copper">·</span> Hog_education_Casandra
              </li>
              <li className="flex gap-3 border-b border-zinc-800 py-2">
                <span className="text-copper">·</span> Hog_education_Donnie
              </li>
              <li className="flex gap-3 border-b border-zinc-800 py-2">
                <span className="text-copper">·</span> Hog_education_Jewel
              </li>
            </ul>
          </div>
          <div className="border border-zinc-800 bg-zinc-950 p-5">
            <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-500">
              Target
            </p>
            <p className="mt-3 font-display text-xl font-bold text-white">
              Next-hour kWh (y<sub className="text-sm">t+1</sub>)
            </p>
            <p className="mt-3 text-sm leading-6 text-zinc-400">
              Chronological 70 / 15 / 15 splits. Weather aligned from BDG2. Occupancy profiles
              are <span className="text-sim">simulated</span> when used as features.
            </p>
            <div className="mt-6 space-y-2">
              {["Train", "Validation", "Test"].map((label, i) => (
                <div key={label} className="flex items-center gap-3">
                  <span className="w-20 font-mono text-[10px] uppercase tracking-wider text-zinc-500">
                    {label}
                  </span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-zinc-900">
                    <motion.div
                      className="h-full rounded-full bg-copper"
                      initial={reduceMotion ? { width: `${[70, 15, 15][i]}%` } : { width: 0 }}
                      whileInView={{ width: `${[70, 15, 15][i]}%` }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.8, delay: 0.15 * i }}
                    />
                  </div>
                  <span className="w-10 text-right font-mono text-xs text-zinc-300">
                    {[70, 15, 15][i]}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Models */}
        <section className="mt-24 border-t border-zinc-800 pt-16">
          <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-copper">
            Models
          </p>
          <h2 className="mt-2 font-display text-2xl font-bold tracking-tight text-white sm:text-3xl">
            Predictions from five forecasters
          </h2>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-zinc-400">
            Same test window for fair comparison. Only trained weights appear on the desk —
            LSTM / CNN-LSTM show “not trained” until you run those jobs.
          </p>
          <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            {MODELS.map((m, i) => (
              <motion.div
                key={m.name}
                className="border border-zinc-800 bg-black px-3 py-4"
                initial={reduceMotion ? false : { opacity: 0, y: 12 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.35, delay: i * 0.06 }}
              >
                <p className="font-mono text-[9px] uppercase tracking-[0.18em] text-zinc-500">
                  {m.kind}
                </p>
                <p className="mt-2 font-display text-sm font-semibold text-white">{m.name}</p>
              </motion.div>
            ))}
          </div>
        </section>

        <footer className="mt-24 border-t border-zinc-800 pt-8">
          <p className="max-w-2xl font-mono text-[11px] leading-5 text-zinc-500">
            Origins stay labeled: BDG2 archive, simulated IoT, or real meter when present. No
            fabricated metrics on this page.
          </p>
          <Link
            to="/login"
            className="mt-6 inline-block bg-copper px-5 py-3 font-display text-sm font-bold text-black hover:brightness-110"
          >
            Enter the desk
          </Link>
        </footer>
      </main>
    </div>
  );
}
