import type { Metadata } from 'next';
import { projects } from '@/data/projects';

export const metadata: Metadata = {
  title: 'Projects',
  description: 'Software engineering projects by Zhaodong Kang.',
};

function ProjectCard({ project }: { project: (typeof projects)[number] }) {
  return (
    <article className="border-b border-stone-200 py-7 first:border-t first:border-stone-200">
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 mb-3">
        <div className="flex items-center gap-3">
          <span className="text-[10px] font-semibold uppercase tracking-widest text-stone-400 border border-stone-200 px-2 py-0.5">
            {project.category}
          </span>
          <span className="text-xs font-mono text-stone-400">{project.year}</span>
        </div>
        {project.link && (
          <a
            href={project.link}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-stone-500 hover:text-stone-900 underline underline-offset-4 transition-colors shrink-0"
          >
            View ↗
          </a>
        )}
      </div>

      <h3 className="font-serif text-xl font-bold text-stone-900 mb-3 leading-snug">
        {project.title}
      </h3>

      <ul className="space-y-1.5 mb-4">
        {project.description.map((line, i) => (
          <li key={i} className="text-sm text-stone-600 leading-relaxed flex gap-2">
            <span className="text-stone-400 shrink-0 mt-1.5 text-xs">▸</span>
            <span>{line}</span>
          </li>
        ))}
      </ul>

      <div className="flex flex-wrap gap-1.5">
        {project.tags.map((tag) => (
          <span key={tag} className="tag text-xs">{tag}</span>
        ))}
      </div>
    </article>
  );
}

export default function ProjectsPage() {
  const sorted = [...projects].sort((a, b) => b.year.localeCompare(a.year));

  return (
    <div className="max-w-5xl mx-auto px-6 py-12 animate-fade-in">

      {/* Page header */}
      <div className="border-b border-stone-200 pb-8 mb-12">
        <h1 className="font-serif text-4xl md:text-5xl font-black text-stone-900 leading-none">
          Projects
        </h1>
      </div>

      <div>
        {sorted.map((project, i) => (
          <ProjectCard key={i} project={project} />
        ))}
      </div>
    </div>
  );
}
