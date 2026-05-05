import { personal } from '@/data/personal';

export default function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className="border-t border-stone-900 mt-20">
      <div className="max-w-5xl mx-auto px-6 py-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-stone-400">
        <span className="font-serif font-bold text-stone-600 tracking-tight">
          {personal.name}
        </span>
        <span>© {year}</span>
      </div>
    </footer>
  );
}
