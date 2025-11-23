/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  images: {
    remotePatterns: [
      { protocol: 'http', hostname: 'localhost', pathname: '/**' },
      { protocol: 'http', hostname: '192.168.1.35', pathname: '/**' },
      // añade hosts que uses
    ],
  },
  // Asegurar que CSS se procese correctamente
  webpack: (config) => {
    config.resolve.fallback = { fs: false, path: false };
    return config;
  },
}
module.exports = nextConfig
