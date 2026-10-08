// Keeping the page responsive while heavy work (a live frame's retouch)
// runs on the main thread: the work goes in tasks of its own, broken up,
// so whatever the user just did is handled — and painted — in between.

/** Resolves in a fresh, ordinary task: input waiting to be handled, and a
 *  frame waiting to be painted, go first. (Not scheduler.yield(): called
 *  from a frame callback, its continuation kept the frame's priority and
 *  starved the page of paints altogether.) */
export function nextTask(): Promise<void> {
  return new Promise((resolve) => {
    const ch = new MessageChannel()
    ch.port1.onmessage = () => resolve()
    ch.port2.postMessage(null)
  })
}
