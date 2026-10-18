// Milestone 3a's figures that need no GPU, computed from scratch: the new op
// sqrt (finite differences, PyTorch goldens, sabotage of its rule and the
// reasons for every miss), and the generator interval limitation found while
// adding it. Deterministic, so CI recomputes them.

import { Graph } from '../ir/graph.js';
import { compareAll } from './golden.js';
import { corpusProgram } from './random-graph.js';
import { sqrtFdCampaign, sqrtGoldenAnalysis } from './analyze-m3-sqrt.js';
import { verifyCpuTraining } from '../train/verify-cpu.js';
import { summarizeM3Gpu } from '../train/summarize-gpu.js';

export const SQRT_FD_PROGRAMS = 400;
export const LIMITATION_PROGRAMS = 2000;

/**
 * log and div coverage of the milestone 1.5 / 2 fuzz corpus, as run and with
 * x*x given its true interval. Both ops are made safe with x*x + c (div's
 * denominator also as -(x*x + c)), so the limitation reaches both (DESIGN.md F1, F6).
 */
function logCoverage(squareAware, f32 = false) {
  let logNodes = 0, programsWithLog = 0, logOfSquarePlusC = 0;
  let divNodes = 0, programsWithDiv = 0, divBySquarePlusC = 0;
  for (let i = 1; i <= LIMITATION_PROGRAMS; i++) {
    const g = corpusProgram(i, { squareAware, f32 }).graph;
    const squarePlusC = (id) => {
      const a = g.nodes[id];
      return a.op === 'add' && g.nodes[a.inputs[0]].op === 'mul' && g.nodes[a.inputs[0]].inputs[0] === g.nodes[a.inputs[0]].inputs[1];
    };
    let hasLog = false, hasDiv = false;
    for (const n of g.nodes) {
      if (n.op === 'log') {
        logNodes++;
        hasLog = true;
        if (squarePlusC(n.inputs[0])) logOfSquarePlusC++;
      } else if (n.op === 'div') {
        divNodes++;
        hasDiv = true;
        const d = g.nodes[n.inputs[1]];
        if (squarePlusC(d.op === 'neg' ? d.inputs[0] : d.id)) divBySquarePlusC++;
      }
    }
    if (hasLog) programsWithLog++;
    if (hasDiv) programsWithDiv++;
  }
  return { logNodes, programsWithLog, logOfSquarePlusC, divNodes, programsWithDiv, divBySquarePlusC };
}

export function measureM3({ sqrtCases, sqrtGolden, trainSpec, torchTrain, optdigits, gpuRecord, gpuDemo }) {
  const sqrt = {
    finiteDifferences: sqrtFdCampaign(SQRT_FD_PROGRAMS),
    golden: { torch: sqrtGolden.torch, ...sqrtGoldenAnalysis(sqrtCases, sqrtGolden, { compareAll, Graph }) },
  };
  const cpuVsTorch = verifyCpuTraining(trainSpec, torchTrain, optdigits);
  return {
    m3: {
      producedBy: 'node tools/measure.js (src/testing/measure-m3.js)',
      sqrt,
      training: {
        cpuVsTorch,
        gpu: gpuRecord ? summarizeM3Gpu({ record: gpuRecord, demo: gpuDemo, spec: trainSpec, torch: torchTrain, sqrt, cpuVsTorch }) : null,
      },
      intervalLimitation: {
        programs: LIMITATION_PROGRAMS,
        corpus: 'the milestone 1.5 and 2 fuzz corpus (corpusProgram 1..N)',
        asRun: logCoverage(false),
        squareAware: logCoverage(true),
        // Milestone 2's GPU fuzzing drew the corpus with the float32 guards on.
        asRunFloat32: logCoverage(false, true),
      },
    },
  };
}
