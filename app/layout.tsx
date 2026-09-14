import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000"),
  title: "PixelProof — Görsel İnceleme",
  description: "E92 ile görsellerdeki yapay zekâ izlerini araştıran, belirsizliği açıkça anlatan yerel öğrenci demosu.",
  openGraph: {
    title: "PixelProof — Görsel İnceleme",
    description: "Fotoğraf seçin, yapay zekâ izlerini araştırın ve sonucun sınırlarını anlayın.",
    type: "website",
    images: [{ url: "/og.png", width: 1536, height: 1024, alt: "PixelProof analiz katmanları" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "PixelProof — Görsel İnceleme",
    description: "Görsellerdeki yapay zekâ izlerini araştıran öğrenci projesi.",
    images: ["/og.png"],
  },
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
