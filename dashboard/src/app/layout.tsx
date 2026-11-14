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
  // Always the desktop layout: phones show it scaled down to fit (pinch to zoom in). Next.js adds initial-scale=1,
  // which scripts/fix-viewport.mjs removes from the built pages (see "postbuild" in package.json).
  width: 1280,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className={`${ui.variable} h-full antialiased`}>
      <body className="min-h-full bg-background">
        {/* Next.js adds a second viewport tag with initial-scale=1 once the page loads, which would open the 1280 px
            page zoomed in on a phone. Keep every viewport tag at a plain width=1280 so phones fit the page instead. */}
        <script dangerouslySetInnerHTML={{ __html: "(function(){var C='width=1280';function fix(){var l=document.querySelectorAll('meta[name=viewport]');for(var i=0;i<l.length;i++){if(l[i].getAttribute('content')!==C)l[i].setAttribute('content',C);}}fix();new MutationObserver(fix).observe(document.head,{childList:true,subtree:true,attributes:true,attributeFilter:['content']});})();" }} />
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  );
}
