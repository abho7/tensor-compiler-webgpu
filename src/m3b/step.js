// Milestone 3b's training step as one IR graph: the transformer's forward pass
// and loss, autodiff's backward pass, and Adam's update (amendments 20 to 25).
//
// Inputs:  every parameter, Adam's state, x and y (one-hot [B·T, V]) and the
//          loss cotangent (always 1).
// Outputs: the loss (from the parameters before the update) and the new value
//          of every parameter and state tensor.

import { grad } from '../autodiff.js';
import { buildForward, paramSpec } from './model.js';
import { adamStateSpec, adamUpdate } from './adam.js';

export const ADAM_3B = { kind: 'adam', lr: 2e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0 };

export const paramNames = (V) => paramSpec(V).map(([n]) => n);
export const paramShapes = (V) => Object.fromEntries(paramSpec(V).map(([n, s]) => [n, s]));

export function buildStep3b(V, B, { bug, rules, opt = ADAM_3B } = {}) {
  const f = buildForward(V, B, { bug });
  const names = paramNames(V);
  const G = grad(f.graph, [f.loss], names.map((n) => f.ids[n]), rules ? { rules } : {});
  const g = G.graph;
  const grads = Object.fromEntries(names.map((n, k) => [n, G.grads[k]]));
  const state = {};
  for (const [name, shape] of adamStateSpec(names, paramShapes(V))) state[name] = g.input(name, shape);
  const outputs = adamUpdate(g, names, f.ids, grads, state, opt);
  return {
    graph: g, names, stateNames: Object.keys(state), loss: f.loss, logits: f.logits, grads,
    inputs: { ...Object.fromEntries(names.map((n) => [n, f.ids[n]])), x: f.ids.x, y: f.ids.y, ...state },
    outputs, cotangent: G.cotangents[0],
  };
}
