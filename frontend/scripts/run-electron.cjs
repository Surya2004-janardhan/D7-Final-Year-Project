#!/usr/bin/env node
const { spawn } = require('child_process');
const path = require('path');

let electronBinary;
try {
  electronBinary = require('electron');
} catch (error) {
  console.error('[electron-runner] Electron is not installed correctly:', error.message);
  process.exit(1);
}

const appRoot = path.join(__dirname, '..');
const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;

const child = spawn(electronBinary, [appRoot], { stdio: 'inherit', env });

child.on('exit', (code) => {
  process.exit(code ?? 0);
});

child.on('error', (error) => {
  console.error('[electron-runner] Failed to start Electron:', error.message);
  process.exit(1);
});
