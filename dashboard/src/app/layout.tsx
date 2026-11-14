import type { Metadata, Viewport } from "next";
import { Plus_Jakarta_Sans } from "next/font/google";

import { AppShell } from "@/components/app-shell";
import { Providers } from "./providers";
import "./globals.css";

// Self-hosted at build time by next/font, so the page makes no runtime font requests.
const ui = Plus_Jakarta_Sans({ variable: "--font-ui", subsets: ["latin"], weight: ["400", "500", "600", "700", "800"], display: "swap" });

export const metadata: Metadata = {
  title: { default: "Atheera Crew Watch", template: "%s | Atheera" },
  description: "Contactless crew health and safety monitoring from the station's Wi-Fi.",
};

export const viewport: Viewport = {
  themeColor: [{ media: "(prefers-color-scheme: light)", color: "#e9eef6" }, { media: "(prefers-color-scheme: dark)", color: "#0f1522" }],
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className={`${ui.variable} h-full antialiased`}>
      <body className="min-h-full bg-background">
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  );
}
