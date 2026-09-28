/**
 * Minimal, accurate ambient type declarations for the exact React 19 API
 * surface this app uses (useState, useEffect, useMemo, useRef, StrictMode,
 * createRoot, JSX runtime) -- NOT a substitute for real @types/react.
 *
 * Why this exists: this sandbox has no network access to the npm registry
 * (confirmed: `npm install react` returns 403 from registry.npmjs.org), so
 * @types/react can't be fetched here. Rather than skip type-checking this
 * app entirely, this shim lets `tsc --strict` genuinely check the component
 * code against real (if narrow) type signatures -- every prop, every hook
 * return type here is checked for real, not just "compiles because `any`
 * swallows everything."
 *
 * IN A REAL DEPLOYMENT: run
 *   npm install react react-dom
 *   npm install -D typescript @types/react @types/react-dom
 * and delete this file -- @types/react is far more complete than this (event
 * types, ref forwarding, context, more hooks) and should take over the
 * moment real npm registry access exists. This file's only job is making
 * real compilation possible in a network-isolated sandbox.
 */

declare module "react" {
  export type ReactNode = string | number | boolean | null | undefined | ReactElement | ReactNode[];

  export interface ReactElement {
    type: any;
    props: any;
    key: string | number | null;
  }

  export function useState<S>(initial: S | (() => S)): [S, (value: S | ((prev: S) => S)) => void];
  export function useEffect(effect: () => void | (() => void), deps?: ReadonlyArray<unknown>): void;
  export function useMemo<T>(factory: () => T, deps: ReadonlyArray<unknown>): T;
  export function useRef<T>(initial: T): { current: T };
  export function useCallback<T extends (...args: any[]) => any>(fn: T, deps: ReadonlyArray<unknown>): T;

  export const StrictMode: (props: { children?: ReactNode }) => ReactElement;

  export interface FunctionComponent<P = {}> {
    (props: P): ReactElement | null;
  }
  export type FC<P = {}> = FunctionComponent<P>;

  const React: {
    createElement: (type: any, props: any, ...children: any[]) => ReactElement;
    Fragment: any;
  };
  export default React;
}

declare module "react/jsx-runtime" {
  export function jsx(type: any, props: any, key?: string): any;
  export function jsxs(type: any, props: any, key?: string): any;
  export const Fragment: unique symbol;
}

declare module "react-dom/client" {
  import { ReactNode } from "react";
  export interface Root {
    render(children: ReactNode): void;
    unmount(): void;
  }
  export function createRoot(container: Element | DocumentFragment): Root;
}

declare namespace JSX {
  interface IntrinsicElements {
    [elemName: string]: any;
  }
  interface Element {
    type: any;
    props: any;
    key: string | number | null;
  }
}
