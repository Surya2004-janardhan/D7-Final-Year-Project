const ipc = typeof window !== 'undefined' && window.require
  ? window.require('electron').ipcRenderer
  : null;

function serialize(payload) {
  if (payload == null) return '';
  try {
    return JSON.stringify(payload);
  } catch {
    return String(payload);
  }
}

function emit(level, scope, message, payload) {
  const text = `[${scope}] ${message}${payload == null ? '' : ` ${serialize(payload)}`}`;
  if (level === 'error') console.error(text);
  else if (level === 'warn') console.warn(text);
  else console.log(text);

  if (ipc) {
    ipc.send('renderer-log', {
      ts: new Date().toISOString(),
      level,
      scope,
      message,
      payload,
    });
  }
}

export function logInfo(scope, message, payload) {
  emit('info', scope, message, payload);
}

export function logWarn(scope, message, payload) {
  emit('warn', scope, message, payload);
}

export function logError(scope, message, payload) {
  emit('error', scope, message, payload);
}
