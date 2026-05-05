/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  // basePath is empty for user GitHub Pages (zhaodong03.github.io)
};

export default nextConfig;
