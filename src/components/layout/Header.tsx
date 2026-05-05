'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';
import ThemeToggle from '@/components/ThemeToggle';

const navLinks = [
  { label: 'About', href: '/' },
  { label: 'CV', href: '/cv/' },
  { label: 'Projects', href: '/projects/' },
  { label: 'Beyond Work', href: '/travels/' },
];

export default function Header() {
  const pathname = usePathname();
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <header className="border-b border-stone-200 bg-stone-50">
      {/* Top bar */}
      <div className="max-w-5xl mx-auto px-6 pt-8 pb-4 relative flex items-start justify-center">
        <div className="absolute right-6 top-2">
          <ThemeToggle />
        </div>
        <Link href="/" className="inline-block group text-center">
          <p className="text-[10px] font-semibold uppercase tracking-[0.3em] text-stone-400 mb-1 group-hover:text-stone-600 transition-colors">
            Portfolio
          </p>
          <h1 className="font-serif text-3xl md:text-4xl font-bold text-stone-900 leading-none">
            Zhaodong Kang
          </h1>
        </Link>
      </div>

      {/* Nav rule */}
      <nav className="border-t border-stone-900">
        <div className="max-w-5xl mx-auto px-6">
          {/* Desktop */}
          <ul className="hidden md:flex items-center justify-center divide-x divide-stone-300">
            {navLinks.map(({ label, href }) => {
              const isActive =
                href === '/' ? pathname === '/' : pathname.startsWith(href);
              return (
                <li key={href}>
                  <Link
                    href={href}
                    className={`block px-6 py-2.5 text-xs font-semibold uppercase tracking-[0.15em] transition-colors ${
                      isActive
                        ? 'bg-stone-900 text-stone-50'
                        : 'text-stone-600 hover:text-stone-900 hover:bg-stone-100'
                    }`}
                  >
                    {label}
                  </Link>
                </li>
              );
            })}
          </ul>

          {/* Mobile */}
          <div className="md:hidden flex items-center justify-between py-2">
            <span className="text-xs font-semibold uppercase tracking-widest text-stone-500">
              Menu
            </span>
            <button
              onClick={() => setMenuOpen(!menuOpen)}
              aria-label="Toggle menu"
              className="p-1 text-stone-600 hover:text-stone-900 transition-colors"
            >
              {menuOpen ? (
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* Mobile dropdown */}
        {menuOpen && (
          <div className="md:hidden border-t border-stone-200 bg-[#faf8f4] dark:bg-[#0c0a09]">
            <ul className="max-w-5xl mx-auto px-6 py-2 flex flex-col">
              {navLinks.map(({ label, href }) => {
                const isActive =
                  href === '/' ? pathname === '/' : pathname.startsWith(href);
                return (
                  <li key={href}>
                    <Link
                      href={href}
                      onClick={() => setMenuOpen(false)}
                      className={`block py-2.5 text-xs font-semibold uppercase tracking-widest border-b border-stone-100 transition-colors ${
                        isActive ? 'text-stone-900' : 'text-stone-500 hover:text-stone-900'
                      }`}
                    >
                      {label}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        )}
      </nav>
    </header>
  );
}
