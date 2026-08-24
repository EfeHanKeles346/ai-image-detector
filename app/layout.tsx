import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PixelProof — Görsel Özgünlük Analizi",
  description: "Fotoğraflardaki gerçek ve yapay zekâ üretimi izlerini inceleyen deneysel analiz arayüzü.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="tr">
      <body>{children}</body>
    </html>
  );
}
