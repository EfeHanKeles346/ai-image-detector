import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PixelProof — E20 Proje Modeli",
  description: "Projede eğitilen E20 ResNet-18 AI görsel modelini çalıştıran ve sınırlarını açıkça gösteren deneysel demo.",
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
