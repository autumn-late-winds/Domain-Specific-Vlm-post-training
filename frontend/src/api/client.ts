const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
    ...options,
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json() as Promise<T>;
}

export type DatasetExample = {
  id: string;
  image: string;
  task_type: string;
  question: string;
  answer: string;
  metadata: Record<string, string>;
};

export type Experiment = {
  experiment_id: string;
  base_model: string;
  training_method: string;
  status: string;
  output_dir: string;
  dataset_size: number;
  validation_size: number;
  mock_notice: string;
};

export type Report = {
  experiment_id: string;
  mode: string;
  mock_notice: string;
  aggregate_metrics: {
    base: Record<string, number>;
    finetuned: Record<string, number>;
    delta_overall: number;
  };
  before_after_examples: Array<{
    id: string;
    task_type: string;
    question: string;
    reference: string;
    base_prediction: string;
    finetuned_prediction: string;
    delta_overall: number;
  }>;
  limitations: string[];
  qualitative_summary: string;
};

export const api = {
  health: () => request<{ status: string; mode: string }>("/health"),
  datasets: () => request<Array<Record<string, string | number>>>("/datasets"),
  stats: () => request<Record<string, unknown>>("/datasets/demo_materials_vlm/stats"),
  validate: () => request<Record<string, unknown>>("/datasets/demo_materials_vlm/validate", { method: "POST" }),
  examples: () => request<DatasetExample[]>("/datasets/demo_materials_vlm/examples?split=val"),
  startTraining: () =>
    request<Experiment>("/training/start", {
      method: "POST",
      body: JSON.stringify({ dataset_id: "demo_materials_vlm", mode: "mock", base_model: "mock-vlm-materials", epochs: 3 }),
    }),
  experiments: () => request<Experiment[]>("/training/experiments"),
  logs: (experimentId: string) => request<Array<Record<string, string | number>>>(`/training/experiments/${experimentId}/logs`),
  runEvaluation: (experimentId: string) =>
    request<Report>("/evaluation/run", {
      method: "POST",
      body: JSON.stringify({ experiment_id: experimentId, dataset_id: "demo_materials_vlm" }),
    }),
  report: (experimentId: string) => request<Report>(`/evaluation/reports/${experimentId}`),
  infer: (question: string, taskType: string, modelVariant: string) =>
    request<{ response: string; mock: boolean }>("/inference/query", {
      method: "POST",
      body: JSON.stringify({ question, task_type: taskType, model_variant: modelVariant }),
    }),
};

export function demoImageUrl(imagePath: string): string {
  return `${API_BASE}/demo/images/${imagePath.split("/").pop()}`;
}
