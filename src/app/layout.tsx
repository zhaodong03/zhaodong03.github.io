import type { Metadata } from 'next';
import './globals.css';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import ThemeProvider from '@/components/ThemeProvider';

export const metadata: Metadata = {
  title: {
    default: 'Zhaodong Kang',
    template: '%s | Zhaodong Kang',
  },
  description:
    'Software Development Engineer at Annapurna Labs (AWS). CS @ Georgia Tech. Passionate about systems, ML, and building things that matter.',
  keywords: ['Zhaodong Kang', 'Software Engineer', 'Georgia Tech', 'AWS', 'Computer Science'],
  authors: [{ name: 'Zhaodong Kang' }],
  openGraph: {
    title: 'Zhaodong Kang',
    description: 'Software Development Engineer at Annapurna Labs (AWS).',
    url: 'https://zhaodong03.github.io',
    siteName: 'Zhaodong Kang',
    locale: 'en_US',
    type: 'website',
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeProvider>
          <Header />
          <main className="min-h-screen">{children}</main>
          <Footer />
        </ThemeProvider>
      </body>
    </html>
  );
}
