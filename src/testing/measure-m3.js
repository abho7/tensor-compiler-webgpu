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

/** log coverage of the milestone 1.5 / 2 fuzz corpus, as run and with x*x given its true interval. */
function logCoverage(squareAware) {
  let logNodes = 0, programsWithLog = 0, logOfSquarePlusC = 0;
  for (let i = 1; i <= LIMITATION_PROGRAMS; i++) {
    const g = corpusProgram(i, { squareAware }).graph;
    let has = false;
    for (const n of g.nodes) {
      if (n.op !== 'log') continue;
      logNodes++;
      has = true;
      const a = g.nodes[n.inputs[0]];
      if (a.op === 'add' && g.nodes[a.inputs[0]].op === 'mul' && g.nodes[a.inputs[0]].inputs[0] === g.nodes[a.inputs[0]].inputs[1]) logOfSquarePlusC++;
    }
    if (has) programsWithLog++;
  }
  return { logNodes, programsWithLog, logOfSquarePlusC };
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
      },
    },
  };
}
