"use client";

import { Component, type ReactNode } from "react";

/** Keeps the dashboard usable if WebGL or a 3D scene fails on this machine. */
export class SceneBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div className="absolute inset-0 grid place-items-center p-6 text-center">
          <div className="grid max-w-sm gap-2">
            <p className="font-bold">The 3D view couldn&apos;t start on this device.</p>
            <p className="text-sm text-muted-foreground">Monitoring, alerts and the crew details still work. Details: {this.state.error.message}</p>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
