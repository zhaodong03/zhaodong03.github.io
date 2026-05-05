import type { Metadata } from 'next';
import { education, workExperience, researchExperience, cvSkills, coursework, type TimelineEntry } from '@/data/cv';

export const metadata: Metadata = {
  title: 'Experience',
  description: 'CV and work experience of Zhaodong Kang — Software Engineer at Annapurna Labs (AWS).',
};

function TimelineItem({ entry, isLast }: { entry: TimelineEntry; isLast: boolean }) {
  return (
    <div className="relative pl-7">
      {/* Vertical line */}
      {!isLast && (
        <div className="absolute left-[5px] top-5 bottom-0 w-px bg-stone-200" />
      )}
      {/* Dot */}
      <div className="absolute left-0 top-1.5 w-2.5 h-2.5 border-2 border-stone-900 bg-stone-50" />

      <div className="pb-10">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-1 mb-2">
          <div>
            <p className="text-base font-bold text-stone-900 leading-snug">
              {entry.link ? (
                <a href={entry.link} target="_blank" rel="noopener noreferrer" className="hover:underline underline-offset-2 transition-colors">
                  {entry.institution}
                </a>
              ) : (
                entry.institution
              )}
            </p>
            <p className="text-sm font-semibold text-stone-700 mt-0.5">{entry.title}</p>
            {entry.location && (
              <p className="text-xs text-stone-400 mt-0.5">{entry.location}</p>
            )}
          </div>
          <span className="text-xs text-stone-400 whitespace-nowrap shrink-0 mt-0.5 font-mono">
            {entry.period}
          </span>
        </div>
        <ul className="space-y-1.5 mt-2.5">
          {entry.description.map((item, i) => (
            <li key={i} className="text-sm text-stone-600 leading-relaxed flex gap-2">
              <span className="text-stone-400 mt-1.5 shrink-0 text-xs">▸</span>
              <span>{item}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default function CVPage() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-12 animate-fade-in">

      {/* Page header */}
      <div className="border-b border-stone-200 pb-8 mb-12">
        <p className="section-heading">Curriculum Vitae</p>
        <h1 className="font-serif text-4xl md:text-5xl font-black text-stone-900 leading-none">
          Experience &<br />Education
        </h1>
      </div>

      <div className="space-y-12">

          {/* Work Experience */}
          <section>
            <p className="section-heading">Work Experience</p>
            <div>
              {workExperience.map((entry, i) => (
                <TimelineItem key={i} entry={entry} isLast={i === workExperience.length - 1} />
              ))}
            </div>
          </section>

          {/* Research Experience */}
          <section>
            <p className="section-heading">Research Experience</p>
            <div>
              {researchExperience.map((entry, i) => (
                <TimelineItem key={i} entry={entry} isLast={i === researchExperience.length - 1} />
              ))}
            </div>
          </section>

          {/* Education */}
          <section>
            <p className="section-heading">Education</p>
            <div>
              {education.map((entry, i) => (
                <TimelineItem key={i} entry={entry} isLast={i === education.length - 1} />
              ))}
            </div>
          </section>

          {/* Skills */}
          <section>
            <p className="section-heading">Skills</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-8 border-t border-stone-200 pt-6">
              {cvSkills.map(({ category, items }) => (
                <div key={category}>
                  <p className="text-[10px] font-semibold uppercase tracking-widest text-stone-400 mb-2">
                    {category}
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {items.map((item) => (
                      <span key={item} className="tag text-xs">{item}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Coursework */}
          <section>
            <p className="section-heading">Coursework</p>
            <div className="divide-y divide-stone-200 border-t border-b border-stone-200">
              {coursework.map(({ semester, courses }) => (
                <div key={semester} className="py-4 flex flex-col sm:flex-row sm:items-start gap-3">
                  <span className="text-xs font-mono text-stone-400 whitespace-nowrap w-28 shrink-0 pt-0.5">
                    {semester}
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {courses.map((c) => (
                      <span key={c} className="tag text-xs">{c}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>
    </div>
  );
}
