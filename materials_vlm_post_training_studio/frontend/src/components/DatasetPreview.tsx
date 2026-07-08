import { DatasetExample, demoImageUrl } from "../api/client";

export default function DatasetPreview({ examples }: { examples: DatasetExample[] }) {
  return (
    <div className="example-grid">
      {examples.map((example) => (
        <article className="example-card" key={example.id}>
          <img src={demoImageUrl(example.image)} alt={example.id} />
          <div>
            <div className="tag-row">
              <span>{example.task_type}</span>
              <span>{example.metadata.modality}</span>
            </div>
            <h3>{example.question}</h3>
            <p>{example.answer}</p>
          </div>
        </article>
      ))}
    </div>
  );
}
