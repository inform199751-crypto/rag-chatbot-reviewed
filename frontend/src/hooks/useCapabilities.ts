import { useEffect, useState } from 'react';
import { type Capabilities, getCapabilities } from '../services/api';

/**
 * What the server behind this UI actually implements.
 *
 * The optimistic default matters: until the answer arrives, assume only the modes that have
 * always worked. Defaulting the other way would flash enabled toggles on load and disable them
 * a moment later, which looks like a bug and invites a click in the gap.
 *
 * A failed request is treated the same as "not supported" rather than surfaced. If the server
 * cannot answer this, the modes it describes are not going to work either, and a toggle that
 * quietly stays off is a better outcome than an error banner about a diagnostic endpoint.
 */
const CONSERVATIVE_DEFAULT: Capabilities = {
  rag: true,
  reasoning: false,
  web_search: false,
  rerank: false,
  answer_without_context: true,
};

export function useCapabilities() {
  const [capabilities, setCapabilities] = useState<Capabilities>(CONSERVATIVE_DEFAULT);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getCapabilities()
      .then((caps) => {
        if (!cancelled) setCapabilities(caps);
      })
      .catch(() => {
        // Keep the conservative default.
      })
      .finally(() => {
        if (!cancelled) setLoaded(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { capabilities, loaded };
}
