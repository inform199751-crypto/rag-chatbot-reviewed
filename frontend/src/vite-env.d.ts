/// <reference types="vite/client" />

// Without this reference `tsc -b` fails on every `import.meta.env` with
// "Property 'env' does not exist on type 'ImportMeta'", because the Vite client types are not
// otherwise in scope. Vite's own dev server strips types and never type-checks, so `yarn dev`
// works while `yarn build` -- which runs `tsc -b` first -- does not. That split is why the
// missing file goes unnoticed until someone tries to produce a production bundle.

interface ImportMetaEnv {
  /** Base URL of the backend. Empty string means same-origin. */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
