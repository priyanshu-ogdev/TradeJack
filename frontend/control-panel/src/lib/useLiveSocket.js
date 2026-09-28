import { useEffect, useRef, useState } from 'react'
import { wsUrl } from './api'

/**
 * Connects to the control panel's /ws/live status stream. Reconnects with
 * exponential backoff (capped) on any close/error -- the same backoff shape used
 * throughout the Python backend's own resilience code (process_supervisor.py,
 * BinanceLiveDepthFeed) -- rather than a bare `new WebSocket(...)` with no
 * recovery, which would leave the panel silently frozen on the last value after
 * any network hiccup or backend restart.
 */
export function useLiveSocket() {
  const [status, setStatus] = useState(null)
  const [connectionState, setConnectionState] = useState('connecting') // connecting | open | closed
  const backoffRef = useRef(1000)
  const closedByUserRef = useRef(false)

  useEffect(() => {
    let socket
    let reconnectTimer

    function connect() {
      setConnectionState('connecting')
      socket = new WebSocket(wsUrl())

      socket.onopen = () => {
        setConnectionState('open')
        backoffRef.current = 1000
      }

      socket.onmessage = (event) => {
        try {
          setStatus(JSON.parse(event.data))
        } catch {
          // A malformed frame should not take down the panel's live view --
          // skip it and wait for the next one.
        }
      }

      socket.onerror = () => {
        socket.close()
      }

      socket.onclose = () => {
        setConnectionState('closed')
        if (closedByUserRef.current) return
        reconnectTimer = setTimeout(connect, backoffRef.current)
        backoffRef.current = Math.min(backoffRef.current * 2, 30000)
      }
    }

    connect()

    return () => {
      closedByUserRef.current = true
      clearTimeout(reconnectTimer)
      socket?.close()
    }
  }, [])

  return { status, connectionState }
}
