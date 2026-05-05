import type { Metadata } from 'next';
import Image from 'next/image';
import Link from 'next/link';
import { personal } from '@/data/personal';
import { experience } from '@/data/cv';

export const metadata: Metadata = {
  title: 'Zhaodong Kang — Software Engineer',
  description: personal.bio,
};

const socialLinks = [
  {
    label: 'GitHub',
    href: `https://github.com/${personal.social.github}`,
    icon: (
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
        <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
      </svg>
    ),
  },
  {
    label: 'LinkedIn',
    href: `https://linkedin.com/in/${personal.social.linkedin}`,
    icon: (
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
      </svg>
    ),
  },
  {
    label: 'Email',
    href: `mailto:${personal.email}`,
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
  },
];

export default function HomePage() {
  const recentExperience = experience.slice(0, 3);

  return (
    <div className="max-w-5xl mx-auto px-6 py-12 animate-fade-in">

      {/* Hero — editorial two-column */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-10 mb-14 pb-14 border-b border-stone-200">
        <div className="md:col-span-2 flex flex-col justify-center">
          <p className="section-heading mb-3">{personal.currentRole}</p>
          <h2 className="font-serif text-5xl md:text-6xl font-black text-stone-900 leading-none mb-6">
            Hello, I&apos;m<br />Zhaodong.
          </h2>
          <p className="text-stone-600 leading-relaxed text-base mb-6 max-w-lg">
            Software engineer passionate about systems, machine learning, and building reliable
            software at scale. Currently working on chip validation at{' '}
            <span className="font-semibold text-stone-900">Annapurna Labs (AWS)</span>,
            and pursuing an M.S. in CS at Georgia Tech.
          </p>
          <div className="flex items-center gap-3 flex-wrap">
            {socialLinks.map(({ label, href, icon }) => (
              <a
                key={label}
                href={href}
                target={label !== 'Email' ? '_blank' : undefined}
                rel="noopener noreferrer"
                aria-label={label}
                className="inline-flex items-center gap-2 px-3 py-1.5 border border-stone-300 text-stone-600 text-xs font-medium hover:border-stone-900 hover:text-stone-900 transition-all"
              >
                {icon}
                {label}
              </a>
            ))}
            <Link
              href="/cv/"
              className="inline-flex items-center gap-1.5 px-4 py-1.5 bg-stone-900 text-stone-50 text-xs font-semibold uppercase tracking-wider hover:bg-stone-700 transition-colors"
            >
              View CV →
            </Link>
          </div>
        </div>

        <div className="flex items-center justify-center md:justify-end">
          <div className="relative w-48 h-48 md:w-56 md:h-56 shrink-0">
            <Image
              src="/assets/img/my_pic.jpg"
              alt="Zhaodong Kang"
              fill
              className="object-cover transition-all duration-500 border border-stone-200"
              priority
            />
          </div>
        </div>
      </section>

      {/* Two-column body */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-12">

        {/* Left: About + Recent Experience */}
        <div className="md:col-span-2 space-y-12">

          {/* About */}
          <section>
            <p className="section-heading">About</p>
            <div className="space-y-4 text-stone-700 leading-relaxed">
              <p>
                I&apos;m a dedicated and enthusiastic computer science student and engineer.
                I hold a <span className="font-semibold text-stone-900">B.S. in Computer Science
                (Intelligence + Information Internetwork)</span> from{' '}
                <a href="https://www.gatech.edu" target="_blank" rel="noopener noreferrer" className="link">
                  Georgia Institute of Technology
                </a>{' '}
                (GPA: 3.96/4.0, Major GPA: 4.0/4.0), with Faculty Honors and Dean&apos;s List recognition.
                I am currently pursuing my M.S. in Computer Science there as well (GPA: 4.0/4.0).
              </p>
              <p>
                I have a strong foundation in algorithms, data structures, OOP, database systems,
                systems &amp; networks, AI, machine learning, and computer vision. I&apos;ve worked
                across the stack — from Python workflow automation at Bank of America to C++ test
                benches for DMA validation in silicon bring-up at AWS.
              </p>
              <p>
                Outside of work, I enjoy traveling, hiking, and photography — visit my{' '}
                <Link href="/travels/" className="link">Travels</Link> page to see where I&apos;ve been.
              </p>
            </div>
          </section>

          {/* Recent Experience */}
          <section>
            <div className="flex items-baseline justify-between mb-5">
              <p className="section-heading mb-0">Recent Experience</p>
              <Link href="/cv/" className="text-xs text-stone-500 hover:text-stone-900 underline underline-offset-4 transition-colors">
                Full CV →
              </Link>
            </div>
            <div className="divide-y divide-stone-200 border-t border-b border-stone-200">
              {recentExperience.map((job, i) => (
                <div key={i} className="py-5">
                  <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-1 mb-2">
                    <div>
                      <p className="font-bold text-stone-900 text-base leading-snug">{job.institution}</p>
                      <p className="text-sm font-semibold text-stone-700 mt-0.5">{job.title}</p>
                      {job.location && <p className="text-xs text-stone-400 mt-0.5">{job.location}</p>}
                    </div>
                    <span className="text-xs text-stone-400 whitespace-nowrap shrink-0">{job.period}</span>
                  </div>
                  <ul className="space-y-1 mt-2">
                    {job.description.map((line, j) => (
                      <li key={j} className="text-sm text-stone-600 leading-relaxed flex gap-2">
                        <span className="text-stone-400 shrink-0 mt-1.5 text-xs">▸</span>
                        <span>{line}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>
        </div>

        {/* Right sidebar: Skills + Contact */}
        <div className="space-y-6">
          <section>
            <p className="section-heading">Skills</p>
            <div className="space-y-4">
              {[
                { label: 'Languages', items: personal.skills.languages },
                { label: 'Frameworks', items: personal.skills.frameworks },
                { label: 'Tools', items: personal.skills.tools },
                { label: 'Areas', items: personal.skills.areas },
              ].map(({ label, items }) => (
                <div key={label}>
                  <p className="text-[10px] font-semibold uppercase tracking-widest text-stone-400 mb-2">{label}</p>
                  <div className="flex flex-wrap gap-1.5">
                    {items.map((item) => (
                      <span key={item} className="tag text-xs">{item}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section>
            <p className="section-heading">Contact</p>
            <div className="space-y-2 text-sm text-stone-600">
              <p>📍 {personal.location}</p>
              <p>
                <a href={`mailto:${personal.email}`} className="link break-all">
                  {personal.email}
                </a>
              </p>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
