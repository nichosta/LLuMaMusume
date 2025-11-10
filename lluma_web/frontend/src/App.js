import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import './App.css';

const socket = io();

function App() {
    const [data, setData] = useState(null);
    const [activeMemoryFile, setActiveMemoryFile] = useState(null);

    useEffect(() => {
        // Fetch initial data
        fetch('/api/latest')
            .then(res => res.json())
            .then(initialData => {
                if (!initialData.error) {
                    setData(initialData);
                }
            });

        // Listen for updates
        socket.on('update', (updateData) => {
            console.log('Received update:', updateData);
            setData(updateData);
        });

        socket.on('connect', () => {
            console.log('Connected to WebSocket server');
        });

        socket.on('disconnect', () => {
            console.log('Disconnected from WebSocket server');
        });

        return () => {
            socket.off('update');
            socket.off('connect');
            socket.off('disconnect');
        };
    }, []);

    const toggleMemoryFile = (filename) => {
        setActiveMemoryFile(activeMemoryFile === filename ? null : filename);
    };

    if (!data) {
        return <div className="loading">Waiting for initial data...</div>;
    }

    return (
        <div className="App">
            <header className="App-header">
                <h1>LLuMa Musume - Agent Monitor</h1>
                <h2>Turn: {data.turn_number}</h2>
            </header>
            <main className="container">
                <div className="left-panel">
                    <section className="agent-reasoning">
                        <h3>Agent Reasoning</h3>
                        <pre>{data.log?.thinking || 'No reasoning available.'}</pre>
                    </section>
                    <section className="tool-calls">
                        <h3>Tool Calls</h3>
                        <pre>{(data.log?.tool_calls && JSON.stringify(data.log.tool_calls, null, 2)) || 'No tool calls.'}</pre>
                    </section>
                </div>
                <div className="right-panel">
                    <section className="screenshot">
                        <h3>Game Screenshot</h3>
                        {data.screenshot_url ? (
                            <img src={data.screenshot_url} alt="Game Screenshot" />
                        ) : (
                            <p>No screenshot available.</p>
                        )}
                    </section>
                    <section className="memory-files">
                        <h3>Memory Files</h3>
                        <ul>
                            {data.memory && Object.keys(data.memory).map(filename => (
                                <li key={filename}>
                                    <button onClick={() => toggleMemoryFile(filename)}>
                                        {filename} {activeMemoryFile === filename ? '▼' : '▶'}
                                    </button>
                                    {activeMemoryFile === filename && (
                                        <pre className="memory-content">
                                            {data.memory[filename]}
                                        </pre>
                                    )}
                                </li>
                            ))}
                        </ul>
                    </section>
                </div>
            </main>
        </div>
    );
}

export default App;
