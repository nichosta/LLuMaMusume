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
                        <h3>Agent Thinking</h3>
                        <pre>{data.log?.agent?.thinking || 'No thinking available.'}</pre>
                    </section>
                    <section className="tool-calls">
                        <h3>Actions Taken</h3>
                        <div className="actions">
                            {data.log?.agent?.memory_actions && data.log.agent.memory_actions.length > 0 && (
                                <div className="memory-actions">
                                    <h4>Memory Files:</h4>
                                    <ul>
                                        {data.log.agent.memory_actions.map((action, idx) => (
                                            <li key={idx}>
                                                <strong>{action.name}</strong>
                                                {action.filename && `: ${action.filename}`}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                            {data.log?.agent?.input_action && (
                                <div className="input-action">
                                    <h4>Button Pressed:</h4>
                                    <p>
                                        <strong>{data.log.agent.input_action.name}</strong>
                                        {data.log.agent.input_action.arguments?.name &&
                                            ` (${data.log.agent.input_action.arguments.name})`}
                                    </p>
                                </div>
                            )}
                            {(!data.log?.agent?.memory_actions || data.log.agent.memory_actions.length === 0) &&
                             !data.log?.agent?.input_action && (
                                <p>No actions taken.</p>
                            )}
                        </div>
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
