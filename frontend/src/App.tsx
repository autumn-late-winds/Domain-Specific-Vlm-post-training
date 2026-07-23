import { useEffect, useMemo, useState } from "react";
import { api, DatasetExample, Experiment, Report } from "./api/client";
import DatasetPreview from "./components/DatasetPreview";
import MetricCard from "./components/MetricCard";

type Tab = "dataset" | "training" | "evaluation" | "playground" | "report";

export default function App() {
  const [tab, setTab] = useState<Tab>("dataset");
  const [health, setHealth] = useState("checking");
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  const [validation, setValidation] = useState<Record<string, unknown> | null>(null);
  const [examples, setExamples] = useState<DatasetExample[]>([]);
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [logs, setLogs] = useState<Array<Record<string, string | number>>>([]);
  const [report, setReport] = useState<Report | null>(null);
  const [busy, setBusy] = useState(false);
  const [question, setQuestion] = useState("Which sample shows the highest performance?");
  const [answer, setAnswer] = useState("");

  useEffect(() => {
    void bootstrap();
  }, []);

  async function bootstrap() {
    const [healthData, statsData, validationData, exampleData, experimentData] = await Promise.all([
      api.health(),
      api.stats(),
      api.validate(),
      api.examples(),
      api.experiments(),
    ]);
    setHealth(`${healthData.status} / ${healthData.mode}`);
    setStats(statsData);
    setValidation(validationData);
    setExamples(exampleData);
    setExperiments(experimentData);
    if (experimentData.length > 0) {
      setExperiment(experimentData[0]);
    }
  }

  async function startMockTraining() {
    setBusy(true);
    try {
      const created = await api.startTraining();
      setExperiment(created);
      setExperiments([created, ...experiments]);
      setLogs(await api.logs(created.experiment_id));
      setTab("evaluation");
    } finally {
      setBusy(false);
    }
  }

  async function runEvaluation() {
    if (!experiment) return;
    setBusy(true);
    try {
      setReport(await api.runEvaluation(experiment.experiment_id));
      setTab("report");
    } finally {
      setBusy(false);
    }
  }

  async function askMockModel(modelVariant: "base" | "finetuned") {
    const result = await api.infer(question, "chart_understanding", modelVariant);
    setAnswer(result.response);
  }

  const selectedExperimentId = useMemo(() => experiment?.experiment_id ?? "none yet", [experiment]);

  return (
    <main>
      <aside className="sidebar">
        <div className="brand-block">
          <span className="mark">VLM</span>
          <h1>Materials VLM Post Training Studio</h1>
          <p>Mock-mode dashboard for materials image QA, captioning, and scientific chart understanding.</p>
        </div>
        <nav>
          {(["dataset", "training", "evaluation", "playground", "report"] as Tab[]).map((item) => (
            <button className={tab === item ? "active" : ""} key={item} onClick={() => setTab(item)}>
              {item}
            </button>
          ))}
        </nav>
        <div className="status-box">
          <span>Backend</span>
          <strong>{health}</strong>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <span className="eyebrow">First-round MVP</span>
            <h2>{titleFor(tab)}</h2>
          </div>
          <div className="experiment-pill">Experiment: {selectedExperimentId}</div>
        </header>

        {tab === "dataset" && (
          <section className="panel-stack">
            <div className="metric-row">
              <MetricCard label="Examples" value={String(stats?.total_examples ?? "-")} hint="train + validation" />
              <MetricCard label="Validation" value={String(examples.length)} hint="before/after eval set" />
              <MetricCard label="Dataset valid" value={validation?.valid ? "yes" : "no"} hint="JSONL schema check" />
            </div>
            <div className="panel">
              <h3>Demo Dataset Preview</h3>
              <DatasetPreview examples={examples} />
            </div>
          </section>
        )}

        {tab === "training" && (
          <section className="panel-grid">
            <div className="panel">
              <h3>Mock Training Config</h3>
              <div className="config-grid">
                <label>mode<input value="mock" readOnly /></label>
                <label>base model<input value="mock-vlm-materials" readOnly /></label>
                <label>epochs<input value="3" readOnly /></label>
                <label>adapter<input value="simulated LoRA metadata" readOnly /></label>
              </div>
              <button className="primary" disabled={busy} onClick={startMockTraining}>
                {busy ? "Running mock training..." : "Start Mock Training"}
              </button>
            </div>
            <div className="panel">
              <h3>Recent Experiments</h3>
              <div className="list">
                {experiments.map((item) => (
                  <button key={item.experiment_id} onClick={() => setExperiment(item)}>
                    <strong>{item.experiment_id}</strong>
                    <span>{item.status} · {item.training_method}</span>
                  </button>
                ))}
              </div>
            </div>
          </section>
        )}

        {tab === "evaluation" && (
          <section className="panel-grid">
            <div className="panel">
              <h3>Before/After Evaluation</h3>
              <p className="muted">Runs the same validation examples through mock base and mock fine-tuned variants.</p>
              <button className="primary" disabled={!experiment || busy} onClick={runEvaluation}>
                {busy ? "Evaluating..." : "Run Evaluation Report"}
              </button>
            </div>
            <div className="panel">
              <h3>Training Logs</h3>
              <pre>{logs.length ? JSON.stringify(logs, null, 2) : "Start or select an experiment to inspect logs."}</pre>
            </div>
          </section>
        )}

        {tab === "playground" && (
          <section className="panel-grid">
            <div className="panel">
              <h3>Inference Playground</h3>
              <textarea value={question} onChange={(event) => setQuestion(event.target.value)} />
              <div className="button-row">
                <button onClick={() => askMockModel("base")}>Ask Base</button>
                <button className="primary" onClick={() => askMockModel("finetuned")}>Ask Fine-tuned</button>
              </div>
            </div>
            <div className="panel">
              <h3>Response</h3>
              <p className="response-text">{answer || "Run a mock query to compare response style."}</p>
            </div>
          </section>
        )}

        {tab === "report" && (
          <section className="panel-stack">
            {report ? (
              <>
                <div className="metric-row">
                  <MetricCard label="Base overall" value={report.aggregate_metrics.base.overall} />
                  <MetricCard label="Fine-tuned overall" value={report.aggregate_metrics.finetuned.overall} />
                  <MetricCard label="Delta" value={report.aggregate_metrics.delta_overall} />
                </div>
                <div className="panel">
                  <h3>Comparison Examples</h3>
                  <div className="comparison-list">
                    {report.before_after_examples.map((row) => (
                      <article key={row.id} className="comparison-card">
                        <span>{row.task_type}</span>
                        <h4>{row.question}</h4>
                        <div className="two-col">
                          <p><strong>Base</strong>{row.base_prediction}</p>
                          <p><strong>Fine-tuned</strong>{row.finetuned_prediction}</p>
                        </div>
                      </article>
                    ))}
                  </div>
                </div>
                <div className="panel">
                  <h3>Limitations</h3>
                  <ul>{report.limitations.map((item) => <li key={item}>{item}</li>)}</ul>
                </div>
              </>
            ) : (
              <div className="panel empty-panel">
                <h3>No report loaded</h3>
                <p>Run mock training and evaluation to create a comparison report.</p>
              </div>
            )}
          </section>
        )}
      </section>
    </main>
  );
}

function titleFor(tab: Tab) {
  const titles: Record<Tab, string> = {
    dataset: "Dataset validation and preview",
    training: "Mock post-training run",
    evaluation: "Model improvement evaluation",
    playground: "Interactive inference playground",
    report: "Exportable evaluation report",
  };
  return titles[tab];
}
