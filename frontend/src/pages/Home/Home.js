import React from 'react'
import { Link } from 'react-router-dom'

import './Home.css'

export default function Home() {

    return (
        <div className='home-container'>
            <div className='home-header'>
                <h1 className='home-heading'>Fitgo Yoga</h1>
                <Link to='/info'>
                    <button 
                        className="btn btn-secondary" 
                        id="about-btn"
                    >
                        info
                    </button>
                </Link>
            </div>

            <h1 className="description">Your personal trainer</h1>
            <div className="home-main">
                <div className="btn-section">
                    <Link to='/start'>
                        <button
                            className="btn start-btn"
                        >Let's Start</button>
                    </Link>
                    <Link to='/tutorials'>
                        <button
                            className="btn start-btn"
                        >Tutorials</button>
                    </Link>

                </div>
            </div>
        </div>
    )
}
