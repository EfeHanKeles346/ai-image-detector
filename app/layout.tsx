import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000"),
  title: "PixelProof — AI Görsel Kanıt Laboratuvarı",
  description: "E26 karar katmanı, E32 R1b araştırma adayı ve E20 taban modelini sınırlarıyla karşılaştıran yerel demo.",
  openGraph: {
    title: "PixelProof — AI Görsel Kanıt Laboratuvarı",
    description: "Ölçülmüş karar ve deneysel model sinyallerini birbirine karıştırmadan karşılaştırın.",
    type: "website",
    images: [{ url: "/og.png", width: 1536, height: 1024, alt: "PixelProof analiz katmanları" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "PixelProof — AI Görsel Kanıt Laboratuvarı",
    description: "Ölçülmüş karar ve deneysel model sinyallerini açık sınırlarıyla karşılaştırın.",
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
