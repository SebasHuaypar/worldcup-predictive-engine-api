import os
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks, Header, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any, Optional
from contextlib import closing
from src.data.database import get_connection, load_all_referees
from src.models.simulator import run_monte_carlo_simulation
from src.data.scraper import run_scraper_pipeline
from src.models.train import train_and_save_models

app = FastAPI(
    title="World Cup Predictive Engine API",
    description="REST API to predict World Cup match outcomes (goals, corners, cards, first goal, exact scores) using Machine Learning and Monte Carlo simulations.",
    version="1.0.0",
    docs_url=None
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS so the user can easily query it from any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")

def verify_admin_key(x_admin_api_key: Optional[str] = Header(None)):
    """Verifies the administrative API key if set in the environment."""
    if ADMIN_API_KEY:
        if not x_admin_api_key or x_admin_api_key != ADMIN_API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Invalid or missing X-Admin-API-Key header."
            )

@app.get("/", tags=["UI"], response_class=HTMLResponse)
def landing_page() -> str:
    """Serves a gorgeous, premium Glassmorphism style dashboard for predictions."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>World Cup Predictive Engine</title>
    <link rel="icon" type="image/webp" href="/static/ball.webp">
    <link href="https://fonts.googleapis.com/css2?family=Anton&family=Oswald:wght@400;600;700;900&family=Noto+Sans:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-color: #030408;
            --card-bg: rgba(10, 12, 22, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-color: #ffffff;
            --text-muted: #9ca3af;
            
            /* Official FWC 26 Colors */
            --fwc-red: #e21a37;
            --fwc-blue: #1a56db;
            --fwc-cyan: #0ea5e9;
            --fwc-green: #16a34a;
            --fwc-yellow: #eab308;
            --fwc-purple: #9333ea;
            
            --primary: var(--fwc-blue);
            --primary-glow: rgba(26, 86, 219, 0.3);
            --accent: var(--fwc-red);
            --accent-glow: rgba(226, 26, 55, 0.3);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Noto Sans', sans-serif;
            scrollbar-width: thin;
            scrollbar-color: rgba(255, 255, 255, 0.4) var(--bg-color);
        }

        /* Custom scrollbar for Webkit browsers (Chrome, Safari, Edge) */
        *::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        *::-webkit-scrollbar-track {
            background: var(--bg-color);
        }

        *::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.4);
            border-radius: 4px;
        }

        *::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.6);
        }

        body {
            background: radial-gradient(circle at top, #0b122b 0%, var(--bg-color) 100%);
            color: var(--text-color);
            min-height: 100vh;
            padding: 2rem 1rem;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
            max-width: 800px;
        }

        .we-are-26-header {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .we-are-26-title {
            font-family: 'Anton', sans-serif;
            font-size: 5.5rem;
            line-height: 0.85;
            display: flex;
            flex-direction: column;
            align-items: center;
            letter-spacing: -2px;
            color: #ffffff;
            margin-bottom: 0.3rem;
            text-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
        }

        .we-are-26-subtitle {
            font-family: 'Oswald', sans-serif;
            font-weight: 900;
            font-size: 1.5rem;
            letter-spacing: 6px;
            color: var(--fwc-cyan);
            text-transform: uppercase;
            border-bottom: 2px solid var(--fwc-red);
            padding-bottom: 4px;
            margin-top: 0.5rem;
        }

        .container {
            width: 100%;
            max-width: 1350px;
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }

        @media (min-width: 900px) {
            .container {
                grid-template-columns: 320px 1fr;
            }
        }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            position: relative;
            overflow: hidden;
        }

        .card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(to right, 
                var(--fwc-red) 0% 16.6%, 
                var(--fwc-blue) 16.6% 33.3%, 
                var(--fwc-cyan) 33.3% 50%, 
                var(--fwc-green) 50% 66.6%, 
                var(--fwc-yellow) 66.6% 83.3%, 
                var(--fwc-purple) 83.3% 100%
            );
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        label {
            font-family: 'Oswald', sans-serif;
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-color);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        select, input {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 0.8rem 1rem;
            color: var(--text-color);
            font-size: 1rem;
            outline: none;
            transition: all 0.3s ease;
            width: 100%;
        }

        select {
            padding-right: 2.5rem;
            appearance: none;
            -webkit-appearance: none;
            -moz-appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 1.2rem center;
            background-size: 1rem;
        }

        select option {
            background-color: #05070f;
            color: var(--text-color);
        }

        select:focus, input:focus {
            border-color: var(--fwc-blue);
            box-shadow: 0 0 10px var(--primary-glow);
            background: rgba(255, 255, 255, 0.06);
        }

        .checkbox-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .checkbox-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.8rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .checkbox-card:hover {
            background: rgba(255, 255, 255, 0.05);
        }

        .checkbox-card input[type="checkbox"] {
            width: auto;
            cursor: pointer;
            accent-color: var(--fwc-red);
        }

        .slider-container {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .slider-container input[type="range"] {
            flex-grow: 1;
            accent-color: var(--fwc-blue);
        }

        .slider-value {
            font-weight: 700;
            min-width: 25px;
            text-align: right;
            color: var(--fwc-cyan);
        }

        button.btn-predict {
            background: linear-gradient(135deg, var(--fwc-red) 0%, #b0152b 100%);
            color: #ffffff;
            border: none;
            border-radius: 6px;
            padding: 1rem;
            font-family: 'Oswald', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(226, 26, 55, 0.35);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-top: 1rem;
        }

        button.btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(226, 26, 55, 0.5);
            background: linear-gradient(135deg, #f43f5e 0%, var(--fwc-red) 100%);
        }

        button.btn-predict:active {
            transform: translateY(0);
        }

        .results-panel {
            display: flex;
            flex-direction: column;
            gap: 2rem;
            width: 100%;
        }

        .placeholder-results {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            min-height: 400px;
            color: var(--text-muted);
            text-align: center;
            border: 2px dashed rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 2rem;
        }

        .placeholder-results svg {
            width: 80px;
            height: 80px;
            margin-bottom: 1.5rem;
            stroke: rgba(255, 255, 255, 0.15);
        }

        .placeholder-results h2 {
            font-family: 'Oswald', sans-serif;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
            color: #ffffff;
        }

        /* Loading Spinner */
        .loading-spinner {
            display: none;
            width: 50px;
            height: 50px;
            border: 5px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border-top-color: var(--fwc-red);
            animation: spin 1s ease-in-out infinite;
            margin: 4rem auto;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Tooltip Styles */
        .tooltip-container {
            display: inline-flex;
            align-items: center;
            position: relative;
            margin-left: 5px;
            cursor: pointer;
            vertical-align: middle;
        }

        .tooltip-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.15);
            color: var(--text-muted);
            font-size: 10px;
            font-weight: 800;
            transition: all 0.2s ease;
        }

        .tooltip-icon:hover {
            background: var(--fwc-cyan);
            color: #05070f;
        }

        .tooltip-text {
            visibility: hidden;
            width: 220px;
            background-color: #0d1330;
            color: #f3f4f6;
            text-align: left;
            border-radius: 8px;
            padding: 8px 12px;
            position: absolute;
            z-index: 10;
            bottom: 125%;
            left: 50%;
            margin-left: -110px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.8rem;
            line-height: 1.3;
            font-weight: 400;
            text-transform: none;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            border: 1px solid var(--border-color);
            pointer-events: none;
        }

        .tooltip-text::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #0d1330 transparent transparent transparent;
        }

        .tooltip-container.right-aligned .tooltip-text {
            left: auto;
            right: 0;
            margin-left: 0;
        }

        .tooltip-container.right-aligned .tooltip-text::after {
            left: auto;
            right: 10px;
            margin-left: 0;
        }

        .tooltip-container:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }

        /* ------------------------------------------------------------------- */
        /* UPGRADED SIMULATION DASHBOARD STYLES */
        /* ------------------------------------------------------------------- */
        
        .dashboard-grid {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            width: 100%;
        }

        .dashboard-row-1 {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }

        .dashboard-row-2 {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }

        @media (min-width: 768px) {
            .dashboard-row-1 {
                grid-template-columns: 1.4fr 1fr;
            }
        }

        @media (min-width: 1024px) {
            .dashboard-row-2 {
                grid-template-columns: 1fr 1fr 1fr;
            }
        }

        .sub-card {
            background: rgba(13, 20, 38, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
            position: relative;
        }

        .sub-card-title {
            font-family: 'Oswald', sans-serif;
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Win Probability Component */
        .win-prob-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            text-align: center;
            padding: 0.5rem 0;
        }

        .win-prob-team {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.4rem;
            flex: 1;
        }

        .win-prob-team-flag {
            height: 32px;
            width: auto;
            border-radius: 3px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 2px 4px rgba(0,0,0,0.4);
            object-fit: cover;
        }

        .win-prob-team-name {
            font-family: 'Oswald', sans-serif;
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--text-color);
            text-transform: uppercase;
        }

        .win-prob-percent {
            font-family: 'Oswald', sans-serif;
            font-size: 2.2rem;
            font-weight: 900;
            line-height: 1;
            margin-top: 0.2rem;
        }

        .win-prob-percent.team-a { color: var(--fwc-cyan); }
        .win-prob-percent.draw { color: var(--text-muted); }
        .win-prob-percent.team-b { color: var(--fwc-red); }

        /* Segmented Bar */
        .segmented-bar {
            height: 10px;
            border-radius: 5px;
            overflow: hidden;
            display: flex;
            background: rgba(255, 255, 255, 0.05);
            margin: 0.5rem 0;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
        }

        .segmented-bar-fill {
            height: 100%;
            transition: width 1s cubic-bezier(0.1, 0.8, 0.2, 1);
        }

        .bar-fill-a { background: linear-gradient(90deg, var(--fwc-blue), var(--fwc-cyan)); }
        .bar-fill-draw { background: #4b5563; }
        .bar-fill-b { background: linear-gradient(90deg, var(--fwc-red), #f43f5e); }

        /* Match Overview Component */
        .overview-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .overview-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px dashed rgba(255, 255, 255, 0.05);
            padding-bottom: 0.6rem;
        }

        .overview-row:last-child {
            border-bottom: none;
            padding-bottom: 0;
        }

        .overview-label {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .overview-value {
            font-family: 'Oswald', sans-serif;
            font-weight: 700;
            font-size: 1.05rem;
            color: var(--text-color);
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        /* Heatmap Grid */
        .matrix-container {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 0.5rem;
            align-items: center;
            margin-top: 0.5rem;
        }

        .matrix-y-label {
            writing-mode: vertical-lr;
            transform: rotate(180deg);
            font-family: 'Oswald', sans-serif;
            font-weight: 700;
            font-size: 0.8rem;
            color: var(--fwc-cyan);
            letter-spacing: 1px;
            text-align: center;
            text-transform: uppercase;
        }

        .matrix-grid-wrapper {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .matrix-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 4px;
            text-align: center;
        }

        .matrix-cell {
            aspect-ratio: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 700;
            transition: all 0.3s ease;
            cursor: pointer;
            border: 1px solid transparent;
            color: var(--text-muted);
        }

        .matrix-cell:hover {
            transform: scale(1.08);
            z-index: 2;
            border-color: rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
        }

        .matrix-header-cell {
            font-family: 'Oswald', sans-serif;
            font-weight: 700;
            font-size: 0.85rem;
            color: var(--text-muted);
            background: transparent !important;
            cursor: default;
        }

        .matrix-header-cell:hover {
            transform: none;
            box-shadow: none;
            border-color: transparent;
        }

        .matrix-x-label {
            font-family: 'Oswald', sans-serif;
            font-weight: 700;
            font-size: 0.8rem;
            color: var(--fwc-purple);
            letter-spacing: 1px;
            text-align: center;
            text-transform: uppercase;
            margin-top: 0.2rem;
        }

        .matrix-legend {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
        }

        .matrix-legend-gradient {
            width: 80px;
            height: 8px;
            border-radius: 4px;
            background: linear-gradient(to right, rgba(38, 47, 85, 0.2), var(--fwc-red));
            border: 1px solid var(--border-color);
        }

        /* Flex Bar Chart for Total Goals */
        .goals-chart-container {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            height: 160px;
            padding: 1rem 0;
            position: relative;
        }

        .goals-chart-column {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.4rem;
            flex: 1;
            height: 100%;
            justify-content: flex-end;
        }

        .goals-bar {
            width: 70%;
            min-width: 15px;
            max-width: 35px;
            border-radius: 6px 6px 0 0;
            background: var(--fwc-purple);
            box-shadow: 0 4px 10px rgba(147, 51, 234, 0.2);
            transition: height 1s cubic-bezier(0.1, 0.8, 0.2, 1);
            height: 0%; /* Set dynamically */
            position: relative;
        }

        .goals-bar-val {
            font-family: 'Oswald', sans-serif;
            font-size: 0.7rem;
            font-weight: 700;
            color: var(--fwc-purple);
            margin-bottom: 0.2rem;
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            white-space: nowrap;
        }

        .goals-label {
            font-family: 'Oswald', sans-serif;
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--text-muted);
        }

        /* Donut Chart styles */
        .donut-chart-wrapper {
            position: relative;
            width: 140px;
            height: 140px;
            margin: 0.5rem auto;
        }

        .donut-center-ball {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            display: flex;
            align-items: center;
            justify-content: center;
            pointer-events: none;
            width: 40px;
            height: 40px;
        }

        .donut-legend {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }

        .donut-legend-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.8rem;
        }

        .donut-legend-info {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .donut-legend-color {
            width: 10px;
            height: 10px;
            border-radius: 2px;
        }

        .donut-legend-color.team-a { background-color: var(--fwc-blue); }
        .donut-legend-color.no-goal { background-color: #4b5563; }
        .donut-legend-color.team-b { background-color: var(--fwc-red); }

        .donut-legend-percent {
            font-family: 'Oswald', sans-serif;
            font-weight: 700;
            color: var(--text-color);
        }

        /* Key Insights Cards */
        .insights-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1rem;
            width: 100%;
        }

        @media (min-width: 600px) {
            .insights-grid {
                grid-template-columns: 1fr 1fr;
            }
        }

        @media (min-width: 1024px) {
            .insights-grid {
                grid-template-columns: repeat(4, 1fr);
            }
        }

        .insight-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            display: flex;
            gap: 0.8rem;
            align-items: flex-start;
            transition: all 0.3s ease;
        }

        .insight-card:hover {
            background: rgba(255, 255, 255, 0.04);
            border-color: rgba(255, 255, 255, 0.12);
            transform: translateY(-2px);
        }

        .insight-icon-container {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .insight-icon-container.favored { background: rgba(234, 179, 8, 0.1); color: var(--fwc-yellow); }
        .insight-icon-container.goals { background: rgba(255, 255, 255, 0.15); color: #ffffff; }
        .insight-icon-container.first-goal { background: rgba(14, 165, 233, 0.1); color: var(--fwc-cyan); }
        .insight-icon-container.btts { background: rgba(22, 163, 74, 0.1); color: var(--fwc-green); }

        .insight-content {
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
        }

        .insight-card-title {
            font-family: 'Oswald', sans-serif;
            font-weight: 700;
            font-size: 0.85rem;
            text-transform: uppercase;
            color: var(--text-color);
        }

        .insight-card-desc {
            font-size: 0.75rem;
            color: var(--text-muted);
            line-height: 1.3;
        }

        /* Export Results Button (Styled exactly like docs OpenAPI Download Button) */
        .btn-export {
            font-family: 'Oswald', sans-serif;
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.8rem;
            text-decoration: none;
            color: #ffffff;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            padding: 0.5rem 0.8rem;
            border-radius: 6px;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s;
            cursor: pointer;
        }

        .btn-export:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .btn-export:active {
            transform: translateY(0);
        }

        .stat-box.ko-details-card {
            background: linear-gradient(135deg, rgba(26, 86, 219, 0.1) 0%, rgba(226, 26, 55, 0.1) 100%);
            border: 1px solid rgba(255, 255, 255, 0.10);
            display: flex;
            flex-direction: column;
            padding: 1.2rem;
            border-radius: 12px;
        }

        .btn-docs-floating {
            position: absolute;
            top: 2rem;
            right: 2rem;
            font-family: 'Oswald', sans-serif;
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.9rem;
            text-decoration: none;
            color: #ffffff;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            padding: 0.6rem 1.2rem;
            border-radius: 6px;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            transition: all 0.2s;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 100;
        }

        .btn-docs-floating:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-1px);
        }

        @media (max-width: 768px) {
            .btn-docs-floating {
                position: static;
                margin: 1rem auto 0 auto;
                width: fit-content;
            }
        }

        /* Custom Alert Modal Styles */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(3, 4, 8, 0.75);
            backdrop-filter: blur(8px);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
            animation: fadeIn 0.2s ease;
        }

        .modal-card {
            background: #0d1222;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 2rem;
            max-width: 400px;
            width: 90%;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
            text-align: center;
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
            position: relative;
            overflow: hidden;
            animation: scaleUp 0.2s ease;
        }
        
        .modal-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(to right, var(--fwc-red), var(--fwc-blue));
        }

        .modal-title {
            font-family: 'Oswald', sans-serif;
            font-size: 1.4rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #ffffff;
        }

        .modal-message {
            font-size: 0.95rem;
            color: var(--text-muted);
            line-height: 1.5;
        }

        .modal-btn {
            background: linear-gradient(135deg, var(--fwc-red) 0%, #b0152b 100%);
            color: #ffffff;
            border: none;
            border-radius: 6px;
            padding: 0.6rem 2rem;
            font-family: 'Oswald', sans-serif;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            text-transform: uppercase;
            align-self: center;
        }

        .modal-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(226, 26, 55, 0.4);
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes scaleUp {
            from { transform: scale(0.95); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }

        /* ------------------------------------------------------------------- */
        /* RESPONSIVE DESIGN ADJUSTMENTS FOR SMALL SCREENS */
        /* ------------------------------------------------------------------- */
        @media (max-width: 600px) {
            body {
                padding: 1rem 0.5rem;
            }

            header {
                margin-bottom: 1.5rem;
            }

            header img {
                max-width: 200px !important;
            }

            header p {
                font-size: 0.95rem !important;
                margin-top: 0.5rem !important;
            }

            .card {
                padding: 1.25rem 1rem !important;
                gap: 1rem !important;
            }

            .sub-card {
                padding: 1rem !important;
                gap: 1rem !important;
            }

            .checkbox-group {
                grid-template-columns: 1fr !important;
                gap: 0.8rem !important;
            }

            .win-prob-percent {
                font-size: 1.8rem !important;
            }

            .win-prob-team-name {
                font-size: 0.85rem !important;
            }

            .win-prob-team-flag {
                height: 26px !important;
            }

            .btn-docs-floating {
                margin: 1rem auto 1.5rem auto !important;
            }
        }

        @media (max-width: 400px) {
            .win-prob-percent {
                font-size: 1.5rem !important;
            }
            
            .win-prob-team-name {
                font-size: 0.75rem !important;
            }
            
            .win-prob-team-flag {
                height: 22px !important;
            }
        }
    </style>
</head>
<body>
    <!-- Floating API Documentation Button -->
    <a href="/docs" class="btn-docs-floating">
        API Docs
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-left: 2px;"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path></svg>
    </a>
    <header>
        <img src="/static/logo.png" alt="World Cup 2026 Logo" style="max-width: 250px; height: auto; margin-bottom: 0.5rem;">
        <p style="font-family: 'Oswald', sans-serif; font-size: 1.1rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; margin-top: 1rem;">
            Advanced ML Engine powered by Optimized Poisson-XGBoost & Monte Carlo Simulations
        </p>
    </header>

    <div class="container">
        <!-- Form Panel -->
        <div class="card" style="padding: 1.5rem; gap: 1.2rem;">
            <div class="form-group">
                <label for="team_a">
                    Team A (Home/Neutral)
                    <span class="tooltip-container">
                        <span class="tooltip-icon">?</span>
                        <span class="tooltip-text">Team designated as home or first neutral team.</span>
                    </span>
                </label>
                <div style="display: flex; gap: 0.6rem; align-items: center;">
                    <img id="flag_a_img" src="https://flagcdn.com/h40/ar.png" alt="Flag" style="height: 22px; width: auto; border-radius: 3px; border: 1px solid var(--border-color); box-shadow: 0 1px 3px rgba(0,0,0,0.5);">
                    <select id="team_a" style="padding: 0.6rem 0.8rem; font-size: 0.95rem;"></select>
                </div>
            </div>
            <div class="form-group">
                <label for="team_b">
                    Team B (Away/Neutral)
                    <span class="tooltip-container">
                        <span class="tooltip-icon">?</span>
                        <span class="tooltip-text">Team designated as away or second neutral team.</span>
                    </span>
                </label>
                <div style="display: flex; gap: 0.6rem; align-items: center;">
                    <img id="flag_b_img" src="https://flagcdn.com/h40/fr.png" alt="Flag" style="height: 22px; width: auto; border-radius: 3px; border: 1px solid var(--border-color); box-shadow: 0 1px 3px rgba(0,0,0,0.5);">
                    <select id="team_b" style="padding: 0.6rem 0.8rem; font-size: 0.95rem;"></select>
                </div>
            </div>

            <div class="checkbox-group">
                <div class="checkbox-card" onclick="document.getElementById('neutral').click(); event.stopPropagation();" style="padding: 0.6rem;">
                    <input type="checkbox" id="neutral" checked onclick="event.stopPropagation();">
                    <label for="neutral" style="cursor:pointer; margin:0; font-size: 0.85rem;">
                        Neutral
                        <span class="tooltip-container">
                            <span class="tooltip-icon">?</span>
                            <span class="tooltip-text">If active, the match is played on neutral ground. Otherwise, Team A gets home advantage.</span>
                        </span>
                    </label>
                </div>
                <div class="checkbox-card" onclick="document.getElementById('knockout').click(); event.stopPropagation();" style="padding: 0.6rem;">
                    <input type="checkbox" id="knockout" onclick="event.stopPropagation();">
                    <label for="knockout" style="cursor:pointer; margin:0; font-size: 0.85rem;">
                        Knockout
                        <span class="tooltip-container">
                            <span class="tooltip-icon">?</span>
                            <span class="tooltip-text">If active, simulates extra time and penalties on a draw to determine who qualifies.</span>
                        </span>
                    </label>
                </div>
            </div>

            <div class="form-group">
                <label style="font-size: 0.85rem;">
                    Team A Absences: <span class="slider-value" id="val_miss_a">0</span>
                    <span class="tooltip-container">
                        <span class="tooltip-icon">?</span>
                        <span class="tooltip-text">Number of missing key players (injury/suspension). Reduces team performance/ELO.</span>
                    </span>
                </label>
                <div class="slider-container">
                    <input type="range" id="missing_players_a" min="0" max="5" value="0" oninput="document.getElementById('val_miss_a').innerText = this.value">
                </div>
            </div>

            <div class="form-group">
                <label style="font-size: 0.85rem;">
                    Team B Absences: <span class="slider-value" id="val_miss_b">0</span>
                    <span class="tooltip-container">
                        <span class="tooltip-icon">?</span>
                        <span class="tooltip-text">Number of missing key players (injury/suspension). Reduces team performance/ELO.</span>
                    </span>
                </label>
                <div class="slider-container">
                    <input type="range" id="missing_players_b" min="0" max="5" value="0" oninput="document.getElementById('val_miss_b').innerText = this.value">
                </div>
            </div>

            <div class="form-group">
                <label for="referee">
                    Referee (Optional)
                    <span class="tooltip-container">
                        <span class="tooltip-icon">?</span>
                        <span class="tooltip-text">Appointed official referee. Statistically scales expected yellow/red cards.</span>
                    </span>
                </label>
                <select id="referee" style="padding: 0.6rem 0.8rem; font-size: 0.95rem;">
                    <option value="">Select referee...</option>
                </select>
            </div>

            <div class="form-group">
                <label for="city">
                    City / Venue (Optional)
                    <span class="tooltip-container">
                        <span class="tooltip-icon">?</span>
                        <span class="tooltip-text">Match city. Altitude affects non-acclimated teams' stamina if above 1500m.</span>
                    </span>
                </label>
                <input type="text" id="city" placeholder="e.g., Mexico City" style="padding: 0.6rem 0.8rem; font-size: 0.95rem;">
            </div>

            <div class="form-group">
                <label for="country">
                    Host Country (Optional)
                    <span class="tooltip-container">
                        <span class="tooltip-icon">?</span>
                        <span class="tooltip-text">Host country of the match. Co-hosts Mexico, Canada, or United States. Activates home ELO boost.</span>
                    </span>
                </label>
                <select id="country" style="padding: 0.6rem 0.8rem; font-size: 0.95rem;">
                    <option value="">None (Pure neutral ground)</option>
                    <option value="Canada">Canada</option>
                    <option value="Mexico">Mexico</option>
                    <option value="United States">United States</option>
                </select>
            </div>

            <div class="form-group">
                <label for="n_sims" style="font-size: 0.85rem;">
                    Simulations: <span class="slider-value" id="val_n_sims">10,000</span>
                    <span class="tooltip-container">
                        <span class="tooltip-icon">?</span>
                        <span class="tooltip-text">Number of matches simulated by the engine to project outcomes. Higher values increase accuracy.</span>
                    </span>
                </label>
                <div class="slider-container">
                    <input type="range" id="n_sims" min="1000" max="100000" step="1000" value="10000" oninput="document.getElementById('val_n_sims').innerText = Number(this.value).toLocaleString('en-US')">
                </div>
            </div>

            <button class="btn-predict" id="btn_simulate" style="padding: 0.8rem; font-size: 1.1rem; margin-top: 0.5rem;">Simulate Match</button>
        </div>

        <!-- Results Panel -->
        <div class="results-panel">
            <div class="loading-spinner" id="spinner"></div>

            <div class="placeholder-results" id="placeholder">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
                <h2>Ready to Simulate</h2>
                <p>Configure teams and click "Simulate Match" to project the outcomes.</p>
            </div>

            <!-- Dynamic Results Card (hidden initially) -->
            <div class="card" id="results_card" style="display: none; width: 100%; gap: 1.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 0.8rem;">
                    <h2 style="font-family: 'Oswald', sans-serif; font-weight: 800; font-size: 1.5rem;" id="results_title">Simulation Results</h2>
                    <button class="btn-export" id="btn_export">
                        Export Results
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                    </button>
                </div>

                <div class="dashboard-grid">
                    <!-- Row 1: Win Probability & Match Overview -->
                    <div class="dashboard-row-1">
                        <!-- Card: Win Probability -->
                        <div class="sub-card">
                            <div class="sub-card-title">
                                Win Probability
                                <span class="tooltip-container right-aligned">
                                    <span class="tooltip-icon">?</span>
                                    <span class="tooltip-text">Monte Carlo projection of match outcome within regular 90 minutes.</span>
                                </span>
                            </div>
                            <div class="win-prob-container">
                                <div class="win-prob-team">
                                    <img id="wp_flag_a" src="https://flagcdn.com/h80/un.png" alt="Flag" class="win-prob-team-flag">
                                    <span class="win-prob-team-name" id="wp_name_a">Team A</span>
                                    <span class="win-prob-percent team-a" id="wp_pct_a">0%</span>
                                </div>
                                <div class="win-prob-team" style="max-width: 80px;">
                                    <span style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); font-weight: bold;">Draw</span>
                                    <span class="win-prob-percent draw" id="wp_pct_draw">0%</span>
                                </div>
                                <div class="win-prob-team">
                                    <img id="wp_flag_b" src="https://flagcdn.com/h80/un.png" alt="Flag" class="win-prob-team-flag">
                                    <span class="win-prob-team-name" id="wp_name_b">Team B</span>
                                    <span class="win-prob-percent team-b" id="wp_pct_b">0%</span>
                                </div>
                            </div>
                            <div class="segmented-bar">
                                <div class="segmented-bar-fill bar-fill-a" id="wp_bar_a" style="width: 0%"></div>
                                <div class="segmented-bar-fill bar-fill-draw" id="wp_bar_draw" style="width: 0%"></div>
                                <div class="segmented-bar-fill bar-fill-b" id="wp_bar_b" style="width: 0%"></div>
                            </div>
                        </div>

                        <!-- Card: Match Overview -->
                        <div class="sub-card">
                            <div class="sub-card-title">
                                Match Overview
                                <span class="tooltip-container right-aligned">
                                    <span class="tooltip-icon">?</span>
                                    <span class="tooltip-text">Key indicators projected from model expectations.</span>
                                </span>
                            </div>
                            <div class="overview-list">
                                <div class="overview-row">
                                    <span class="overview-label">Most Likely Score</span>
                                    <span class="overview-value" id="ov_score" style="color: var(--fwc-cyan)">0 - 0</span>
                                </div>
                                <div class="overview-row">
                                    <span class="overview-label">Expected Goals (xG)</span>
                                    <span class="overview-value" id="ov_xg">0.00 - 0.00</span>
                                </div>
                                <div class="overview-row">
                                    <span class="overview-label">Both Teams to Score</span>
                                    <span class="overview-value" id="ov_btts">0%</span>
                                </div>
                                <div class="overview-row">
                                    <span class="overview-label">First Goal</span>
                                    <span class="overview-value" id="ov_first_goal">None</span>
                                </div>
                                <div class="overview-row">
                                    <span class="overview-label">Avg. Total Goals</span>
                                    <span class="overview-value" id="ov_total_goals">0.00</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Knockout Details Card (hidden by default) -->
                    <div class="stat-box ko-details-card" id="ko_card" style="display: none;">
                        <span class="stat-box-title" style="color: #67e8f9;">Knockout Stage Progression (To Qualify)</span>
                        <span class="stat-box-value" style="font-size: 1.5rem; margin-top: 0.5rem;" id="val_qualify"></span>
                    </div>

                    <!-- Row 2: Probability Matrix & Distribution & First Goal -->
                    <div class="dashboard-row-2">
                        <!-- Card: Score Probability Matrix -->
                        <div class="sub-card">
                            <div class="sub-card-title">
                                Score Probability Matrix
                                <span class="tooltip-container right-aligned">
                                    <span class="tooltip-icon">?</span>
                                    <span class="tooltip-text">Poisson-distributed score frequency grid. Highlight shows most likely score.</span>
                                </span>
                            </div>
                            <div class="matrix-container">
                                <div class="matrix-y-label" id="matrix_y_label">A</div>
                                <div class="matrix-grid-wrapper">
                                    <div class="matrix-grid" id="matrix_grid">
                                        <!-- Dynamically generated -->
                                    </div>
                                    <div class="matrix-x-label" id="matrix_x_label">B</div>
                                </div>
                            </div>
                            <div class="matrix-legend">
                                <span>Lower</span>
                                <div class="matrix-legend-gradient"></div>
                                <span>Higher</span>
                            </div>
                        </div>

                        <!-- Card: Total Goals Distribution -->
                        <div class="sub-card">
                            <div class="sub-card-title">
                                Total Goals Distribution
                                <span class="tooltip-container right-aligned">
                                    <span class="tooltip-icon">?</span>
                                    <span class="tooltip-text">Probability distribution of combined goals scored in 90 minutes.</span>
                                </span>
                            </div>
                            <div class="goals-chart-container" id="goals_chart">
                                <!-- Dynamically generated columns -->
                            </div>
                        </div>

                        <!-- Card: First Goal Probability -->
                        <div class="sub-card">
                            <div class="sub-card-title">
                                First Goal Probability
                                <span class="tooltip-container right-aligned">
                                    <span class="tooltip-icon">?</span>
                                    <span class="tooltip-text">Likelihood of which team scores the opening goal.</span>
                                </span>
                            </div>
                            <div class="donut-chart-wrapper">
                                <canvas id="first_goal_chart"></canvas>
                                <div class="donut-center-ball">
                                    <img src="/static/ball.webp" alt="Soccer Ball" style="width: 40px; height: 40px; object-fit: contain;">
                                </div>
                            </div>
                            <div class="donut-legend">
                                <div class="donut-legend-item">
                                    <div class="donut-legend-info">
                                        <div class="donut-legend-color team-a"></div>
                                        <span id="fg_name_a">Team A</span>
                                    </div>
                                    <span class="donut-legend-percent" id="fg_pct_a">0%</span>
                                </div>
                                <div class="donut-legend-item">
                                    <div class="donut-legend-info">
                                        <div class="donut-legend-color no-goal"></div>
                                        <span>No Goal</span>
                                    </div>
                                    <span class="donut-legend-percent" id="fg_pct_no">0%</span>
                                </div>
                                <div class="donut-legend-item">
                                    <div class="donut-legend-info">
                                        <div class="donut-legend-color team-b"></div>
                                        <span id="fg_name_b">Team B</span>
                                    </div>
                                    <span class="donut-legend-percent" id="fg_pct_b">0%</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Row 3: Key Insights -->
                    <div class="sub-card">
                        <div class="sub-card-title">
                            Key Match Insights
                            <span class="tooltip-container right-aligned">
                                <span class="tooltip-icon">?</span>
                                <span class="tooltip-text">Automated logical insights derived from the model simulation outputs.</span>
                            </span>
                        </div>
                        <div class="insights-grid">
                            <!-- Insight 1: Favored -->
                            <div class="insight-card">
                                <div class="insight-icon-container favored">
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
                                </div>
                                <div class="insight-content">
                                    <span class="insight-card-title" id="ins_title_favored">Match Favored</span>
                                    <span class="insight-card-desc" id="ins_desc_favored">Calculating win probabilities...</span>
                                </div>
                            </div>
                            <!-- Insight 2: High/Low Scoring -->
                            <div class="insight-card">
                                <div class="insight-icon-container goals">
                                    <img src="/static/ball.webp" alt="Goals" style="width: 20px; height: 20px; object-fit: contain;">
                                </div>
                                <div class="insight-content">
                                    <span class="insight-card-title" id="ins_title_goals">Scoring Expectations</span>
                                    <span class="insight-card-desc" id="ins_desc_goals">Analyzing expected goals...</span>
                                </div>
                            </div>
                            <!-- Insight 3: First Goal -->
                            <div class="insight-card">
                                <div class="insight-icon-container first-goal">
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>
                                </div>
                                <div class="insight-content">
                                    <span class="insight-card-title" id="ins_title_first_goal">First Goal Advantage</span>
                                    <span class="insight-card-desc" id="ins_desc_first_goal">Calculating opening probabilities...</span>
                                </div>
                            </div>
                            <!-- Insight 4: BTTS -->
                            <div class="insight-card">
                                <div class="insight-icon-container btts">
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                                </div>
                                <div class="insight-content">
                                    <span class="insight-card-title" id="ins_title_btts">Both Teams to Score</span>
                                    <span class="insight-card-desc" id="ins_desc_btts">Calculating clean sheet metrics...</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const API_URL = ""; // Relative URL so it works on any host

        // 48 teams of World Cup 2026 and flags
        const WORLD_CUP_2026_TEAMS = [
            { name: "Algeria", flag: "🇩🇿" },
            { name: "Argentina", flag: "🇦🇷" },
            { name: "Australia", flag: "🇦🇺" },
            { name: "Austria", flag: "🇦🇹" },
            { name: "Belgium", flag: "🇧🇪" },
            { name: "Bosnia and Herzegovina", flag: "🇧🇦" },
            { name: "Brazil", flag: "🇧🇷" },
            { name: "Canada", flag: "🇨🇦" },
            { name: "Cape Verde", flag: "🇨🇻" },
            { name: "Colombia", flag: "🇨🇴" },
            { name: "Croatia", flag: "🇭🇷" },
            { name: "Curaçao", flag: "🇨🇼" },
            { name: "Czech Republic", flag: "🇨🇿" },
            { name: "DR Congo", flag: "🇨🇩" },
            { name: "Ecuador", flag: "🇪🇨" },
            { name: "Egypt", flag: "🇪🇬" },
            { name: "England", flag: "🏴󠁧󠁢󠁥󠁮󠁧󠁿" },
            { name: "France", flag: "🇫🇷" },
            { name: "Germany", flag: "🇩🇪" },
            { name: "Ghana", flag: "🇬🇭" },
            { name: "Haiti", flag: "🇭🇹" },
            { name: "Iran", flag: "🇮🇷" },
            { name: "Iraq", flag: "🇮🇶" },
            { name: "Ivory Coast", flag: "🇨🇮" },
            { name: "Japan", flag: "🇯🇵" },
            { name: "Jordan", flag: "🇯🇴" },
            { name: "Mexico", flag: "🇲🇽" },
            { name: "Morocco", flag: "🇲🇦" },
            { name: "Netherlands", flag: "🇳🇱" },
            { name: "New Zealand", flag: "🇳🇿" },
            { name: "Norway", flag: "🇳🇴" },
            { name: "Panama", flag: "🇵🇦" },
            { name: "Paraguay", flag: "🇵🇾" },
            { name: "Portugal", flag: "🇵🇹" },
            { name: "Qatar", flag: "🇶🇦" },
            { name: "Saudi Arabia", flag: "🇸🇦" },
            { name: "Scotland", flag: "🏴󠁧󠁢󠁳󠁣󠁴󠁿" },
            { name: "Senegal", flag: "🇸🇳" },
            { name: "South Africa", flag: "🇿🇦" },
            { name: "South Korea", flag: "🇰🇷" },
            { name: "Spain", flag: "🇪🇸" },
            { name: "Sweden", flag: "🇸🇪" },
            { name: "Switzerland", flag: "🇨🇭" },
            { name: "Tunisia", flag: "🇹🇳" },
            { name: "Turkey", flag: "🇹🇷" },
            { name: "United States", flag: "🇺🇸" },
            { name: "Uruguay", flag: "🇺🇾" },
            { name: "Uzbekistan", flag: "🇺🇿" }
        ];

        const ISO_COUNTRY_CODES = {
            "Algeria": "dz",
            "Argentina": "ar",
            "Australia": "au",
            "Austria": "at",
            "Belgium": "be",
            "Bosnia and Herzegovina": "ba",
            "Brazil": "br",
            "Canada": "ca",
            "Cape Verde": "cv",
            "Colombia": "co",
            "Croatia": "hr",
            "Curaçao": "cw",
            "Czech Republic": "cz",
            "DR Congo": "cd",
            "Ecuador": "ec",
            "Egypt": "eg",
            "England": "gb-eng",
            "France": "fr",
            "Germany": "de",
            "Ghana": "gh",
            "Haiti": "ht",
            "Iran": "ir",
            "Iraq": "iq",
            "Ivory Coast": "ci",
            "Japan": "jp",
            "Jordan": "jo",
            "Mexico": "mx",
            "Morocco": "ma",
            "Netherlands": "nl",
            "New Zealand": "nz",
            "Norway": "no",
            "Panama": "pa",
            "Paraguay": "py",
            "Portugal": "pt",
            "Qatar": "qa",
            "Saudi Arabia": "sa",
            "Scotland": "gb-sct",
            "Senegal": "sn",
            "South Africa": "za",
            "South Korea": "kr",
            "Spain": "es",
            "Sweden": "se",
            "Switzerland": "ch",
            "Tunisia": "tn",
            "Turkey": "tr",
            "United States": "us",
            "Uruguay": "uy",
            "Uzbekistan": "uz"
        };

        function getTeamAbbr(name) {
            if (!name) return "";
            const clean = name.toLowerCase();
            if (clean === "united states") return "USA";
            if (clean === "saudi arabia") return "KSA";
            if (clean === "south korea") return "KOR";
            if (clean === "new zealand") return "NZL";
            if (clean === "south africa") return "RSA";
            if (clean === "dr congo") return "COD";
            if (clean === "cape verde") return "CPV";
            if (clean === "ivory coast") return "CIV";
            if (clean === "czech republic") return "CZE";
            if (clean === "bosnia and herzegovina") return "BIH";
            return name.substring(0, 3).toUpperCase();
        }

        let lastPredictionData = null;

        // Load data on page load
        window.addEventListener('DOMContentLoaded', async () => {
            try {
                // Populate 2026 World Cup teams statically
                const selectA = document.getElementById('team_a');
                const selectB = document.getElementById('team_b');
                
                selectA.innerHTML = '';
                selectB.innerHTML = '';
                
                WORLD_CUP_2026_TEAMS.forEach(team => {
                    const optA = new Option(team.name, team.name);
                    const optB = new Option(team.name, team.name);
                    selectA.add(optA);
                    selectB.add(optB);
                });

                // Set default selections
                selectA.value = "Argentina";
                selectB.value = "France";
                
                // Initialize default flag images adjacent to dropdowns
                document.getElementById('flag_a_img').src = "https://flagcdn.com/h40/ar.png";
                document.getElementById('flag_b_img').src = "https://flagcdn.com/h40/fr.png";

                // Fetch Referees
                const resRefs = await fetch(API_URL + '/api/v1/referees');
                const dataRefs = await resRefs.json();
                const selectRef = document.getElementById('referee');
                
                dataRefs.referees.forEach(ref => {
                    const opt = new Option(ref.referee, ref.referee);
                    selectRef.add(opt);
                });
            } catch (err) {
                console.error("Error loading initial data:", err);
            }
        });

        // Dropdown change handlers to update adjacent flags
        document.getElementById('team_a').addEventListener('change', (e) => {
            const code = ISO_COUNTRY_CODES[e.target.value] || "un";
            document.getElementById('flag_a_img').src = `https://flagcdn.com/h40/${code}.png`;
        });

        document.getElementById('team_b').addEventListener('change', (e) => {
            const code = ISO_COUNTRY_CODES[e.target.value] || "un";
            document.getElementById('flag_b_img').src = `https://flagcdn.com/h40/${code}.png`;
        });

        // Export Results event listener
        document.getElementById('btn_export').addEventListener('click', () => {
            if (!lastPredictionData) return;
            const teamA = lastPredictionData.match_info.team_a;
            const teamB = lastPredictionData.match_info.team_b;
            
            const jsonString = JSON.stringify(lastPredictionData, null, 2);
            const blob = new Blob([jsonString], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `prediction_${teamA.toLowerCase().replace(/\\s+/g, '_')}_vs_${teamB.toLowerCase().replace(/\\s+/g, '_')}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });

        // Trigger simulation
        document.getElementById('btn_simulate').addEventListener('click', async () => {
            const teamA = document.getElementById('team_a').value;
            const teamB = document.getElementById('team_b').value;
            const neutral = document.getElementById('neutral').checked;
            const knockout = document.getElementById('knockout').checked;
            const missA = document.getElementById('missing_players_a').value;
            const missB = document.getElementById('missing_players_b').value;
            const referee = document.getElementById('referee').value;
            const city = document.getElementById('city').value;
            const country = document.getElementById('country').value;
            const nSims = document.getElementById('n_sims').value;

            // Basic validation
            if (teamA === teamB) {
                showAlert("Please select two different teams.");
                return;
            }

            // Show Spinner and Hide Placeholder
            document.getElementById('placeholder').style.display = 'none';
            document.getElementById('results_card').style.display = 'none';
            document.getElementById('spinner').style.display = 'block';

            try {
                let url = `${API_URL}/api/v1/predict?team_a=${encodeURIComponent(teamA)}&team_b=${encodeURIComponent(teamB)}&neutral=${neutral}&knockout=${knockout}&missing_players_a=${missA}&missing_players_b=${missB}&n_sims=${nSims}`;
                if (referee) url += `&referee=${encodeURIComponent(referee)}`;
                if (city) url += `&city=${encodeURIComponent(city)}`;
                if (country) url += `&country=${encodeURIComponent(country)}`;

                const res = await fetch(url);
                const data = await res.json();
                
                // Store results for exporting
                lastPredictionData = data;
                
                document.getElementById('spinner').style.display = 'none';
                document.getElementById('results_card').style.display = 'flex';

                // Render match title
                document.getElementById('results_title').innerText = `Simulation Results: ${teamA} vs ${teamB}`;

                // Update Outcome Bars
                const pWinA = data.predictions.outcomes.win_a;
                const pDraw = data.predictions.outcomes.draw;
                const pWinB = data.predictions.outcomes.win_b;

                // Win Probability Text and SVG flags
                const codeA = ISO_COUNTRY_CODES[teamA] || "un";
                const codeB = ISO_COUNTRY_CODES[teamB] || "un";

                document.getElementById('wp_flag_a').src = `https://flagcdn.com/h80/${codeA}.png`;
                document.getElementById('wp_name_a').innerText = teamA;
                document.getElementById('wp_pct_a').innerText = `${(pWinA * 100).toFixed(1)}%`;

                document.getElementById('wp_pct_draw').innerText = `${(pDraw * 100).toFixed(1)}%`;

                document.getElementById('wp_flag_b').src = `https://flagcdn.com/h80/${codeB}.png`;
                document.getElementById('wp_name_b').innerText = teamB;
                document.getElementById('wp_pct_b').innerText = `${(pWinB * 100).toFixed(1)}%`;

                // Set widths of segmented bar
                document.getElementById('wp_bar_a').style.width = `${pWinA * 100}%`;
                document.getElementById('wp_bar_draw').style.width = `${pDraw * 100}%`;
                document.getElementById('wp_bar_b').style.width = `${pWinB * 100}%`;

                // Expected Goals, Corners, Cards
                const expGoalsA = data.predictions.goals.expected_a;
                const expGoalsB = data.predictions.goals.expected_b;
                const bttsProb = (1 - Math.exp(-expGoalsA)) * (1 - Math.exp(-expGoalsB));

                // Match Overview List
                document.getElementById('ov_score').innerText = data.predictions.goals.top_exact_scores[0].score.replace('-', ' - ');
                document.getElementById('ov_xg').innerText = `${expGoalsA.toFixed(2)} - ${expGoalsB.toFixed(2)}`;
                document.getElementById('ov_btts').innerText = `${(bttsProb * 100).toFixed(1)}%`;

                const fgA = data.predictions.events.first_goal_team_a;
                const fgB = data.predictions.events.first_goal_team_b;
                const fgNo = data.predictions.events.no_goals;

                let firstGoalMarkup = '';
                if (fgA >= fgB && fgA >= fgNo) {
                    firstGoalMarkup = `<img src="https://flagcdn.com/h24/${codeA}.png" style="height: 14px; width: auto; border-radius: 2px; border: 1px solid rgba(255,255,255,0.1); margin-right: 4px; vertical-align: middle;"> ${(fgA * 100).toFixed(1)}%`;
                } else if (fgB >= fgA && fgB >= fgNo) {
                    firstGoalMarkup = `<img src="https://flagcdn.com/h24/${codeB}.png" style="height: 14px; width: auto; border-radius: 2px; border: 1px solid rgba(255,255,255,0.1); margin-right: 4px; vertical-align: middle;"> ${(fgB * 100).toFixed(1)}%`;
                } else {
                    firstGoalMarkup = `No Goal ${(fgNo * 100).toFixed(1)}%`;
                }
                document.getElementById('ov_first_goal').innerHTML = firstGoalMarkup;
                document.getElementById('ov_total_goals').innerText = data.predictions.goals.expected_total.toFixed(2);

                // Update Knockout details
                const koCard = document.getElementById('ko_card');
                if (data.predictions.knockout_details) {
                    koCard.style.display = 'flex';
                    const qualA = data.predictions.knockout_details.qualify_a;
                    const qualB = data.predictions.knockout_details.qualify_b;
                    document.getElementById('val_qualify').innerHTML = `
                        <img src="https://flagcdn.com/h24/${codeA}.png" style="height: 16px; width: auto; border-radius: 2px; vertical-align: middle; margin-right: 4px;"> 
                        ${teamA} ${(qualA * 100).toFixed(1)}% 
                        <span style="color: var(--text-muted); margin: 0 0.8rem;">|</span> 
                        ${(qualB * 100).toFixed(1)}% ${teamB} 
                        <img src="https://flagcdn.com/h24/${codeB}.png" style="height: 16px; width: auto; border-radius: 2px; vertical-align: middle; margin-left: 4px;">
                    `;
                } else {
                    koCard.style.display = 'none';
                }

                // -------------------------------------------------------------
                // 1. GENERATE SCORE PROBABILITY MATRIX HEATMAP (5x5 GRID)
                // -------------------------------------------------------------
                const matrixGrid = document.getElementById('matrix_grid');
                matrixGrid.innerHTML = '';

                document.getElementById('matrix_y_label').innerText = getTeamAbbr(teamA);
                document.getElementById('matrix_x_label').innerText = getTeamAbbr(teamB);

                const pA = [];
                const pB = [];

                function poisson(lambda, k) {
                    let fact = 1;
                    for (let i = 2; i <= k; i++) fact *= i;
                    return Math.exp(-lambda) * Math.pow(lambda, k) / fact;
                }

                let sumA = 0;
                let sumB = 0;
                for (let g = 0; g < 4; g++) {
                    const probA = poisson(expGoalsA, g);
                    const probB = poisson(expGoalsB, g);
                    pA.push(probA);
                    pB.push(probB);
                    sumA += probA;
                    sumB += probB;
                }
                pA.push(Math.max(0, 1 - sumA));
                pB.push(Math.max(0, 1 - sumB));

                // Find highest probability cell coordinates corresponding to top exact score
                const topScoreStr = data.predictions.goals.top_exact_scores[0].score;
                let topScoreRow = -1;
                let topScoreCol = -1;
                if (topScoreStr) {
                    const parts = topScoreStr.split('-');
                    const valA = parseInt(parts[0]);
                    const valB = parseInt(parts[1]);
                    topScoreRow = valA >= 4 ? 4 : valA;
                    topScoreCol = valB >= 4 ? 4 : valB;
                }

                // Empty top-left header cell
                const emptyCell = document.createElement('div');
                emptyCell.className = 'matrix-cell matrix-header-cell';
                matrixGrid.appendChild(emptyCell);

                // Column headers (Team B goals)
                for (let col = 0; col < 5; col++) {
                    const headerCell = document.createElement('div');
                    headerCell.className = 'matrix-cell matrix-header-cell';
                    headerCell.innerText = col === 4 ? '4+' : col;
                    matrixGrid.appendChild(headerCell);
                }

                let maxCellProb = 0;
                const cellProbs = [];
                for (let r = 0; r < 5; r++) {
                    cellProbs[r] = [];
                    for (let c = 0; c < 5; c++) {
                        const val = pA[r] * pB[c];
                        cellProbs[r][c] = val;
                        if (val > maxCellProb) maxCellProb = val;
                    }
                }

                // Fill rows
                for (let r = 0; r < 5; r++) {
                    // Row header cell (Team A goals)
                    const rowHeader = document.createElement('div');
                    rowHeader.className = 'matrix-cell matrix-header-cell';
                    rowHeader.innerText = r === 4 ? '4+' : r;
                    matrixGrid.appendChild(rowHeader);

                    for (let c = 0; c < 5; c++) {
                        const cell = document.createElement('div');
                        cell.className = 'matrix-cell';
                        const val = cellProbs[r][c];
                        cell.innerText = `${(val * 100).toFixed(1)}%`;
                        
                        const isTop = (r === topScoreRow && c === topScoreCol);
                        if (isTop) {
                            cell.style.background = 'var(--fwc-red)';
                            cell.style.color = '#ffffff';
                            cell.style.fontWeight = '900';
                            cell.style.borderColor = 'rgba(255, 255, 255, 0.4)';
                            cell.style.boxShadow = '0 0 12px rgba(226, 26, 55, 0.5)';
                        } else {
                            const ratio = maxCellProb > 0 ? (val / maxCellProb) : 0;
                            cell.style.background = `rgba(38, 47, 85, ${0.1 + ratio * 0.75})`;
                            cell.style.color = ratio > 0.4 ? '#ffffff' : 'var(--text-muted)';
                        }
                        matrixGrid.appendChild(cell);
                    }
                }

                // -------------------------------------------------------------
                // 2. GENERATE TOTAL GOALS DISTRIBUTION CHART
                // -------------------------------------------------------------
                const goalsChart = document.getElementById('goals_chart');
                goalsChart.innerHTML = '';

                const lambdaTotal = expGoalsA + expGoalsB;
                const totalProbs = [];
                let totalSum = 0;
                for (let k = 0; k < 6; k++) {
                    const prob = poisson(lambdaTotal, k);
                    totalProbs.push(prob);
                    totalSum += prob;
                }
                totalProbs.push(Math.max(0, 1 - totalSum)); // 6+ goals

                const maxGoalProb = Math.max(...totalProbs);

                for (let k = 0; k < 7; k++) {
                    const prob = totalProbs[k];
                    const pct = (prob * 100).toFixed(1);
                    const ratio = maxGoalProb > 0 ? (prob / maxGoalProb) : 0;
                    const barHeight = Math.max(5, ratio * 75); // scale between 5% and 75% height

                    const colDiv = document.createElement('div');
                    colDiv.className = 'goals-chart-column';

                    colDiv.innerHTML = `
                        <div class="goals-bar" style="height: ${barHeight}%">
                            <span class="goals-bar-val">${pct}%</span>
                        </div>
                        <span class="goals-label">${k === 6 ? '6+' : k}</span>
                    `;
                    goalsChart.appendChild(colDiv);
                }

                // -------------------------------------------------------------
                // 3. GENERATE FIRST GOAL PROBABILITY DONUT CHART
                // -------------------------------------------------------------
                if (window.firstGoalChartObj) {
                    window.firstGoalChartObj.destroy();
                }

                document.getElementById('fg_name_a').innerText = teamA;
                document.getElementById('fg_name_b').innerText = teamB;
                document.getElementById('fg_pct_a').innerText = `${(fgA * 100).toFixed(1)}%`;
                document.getElementById('fg_pct_no').innerText = `${(fgNo * 100).toFixed(1)}%`;
                document.getElementById('fg_pct_b').innerText = `${(fgB * 100).toFixed(1)}%`;

                // Register custom tooltip positioner to follow mouse cursor
                if (typeof Chart !== 'undefined' && Chart.Tooltip && Chart.Tooltip.positioners) {
                    Chart.Tooltip.positioners.followCursor = function(elements, eventPosition) {
                        const pos = eventPosition && typeof eventPosition.x === 'number' ? eventPosition : null;
                        return pos ? { x: pos.x, y: pos.y } : Chart.Tooltip.positioners.average(elements);
                    };
                }

                const ctx = document.getElementById('first_goal_chart').getContext('2d');
                window.firstGoalChartObj = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: [teamA, 'No Goal', teamB],
                        datasets: [{
                            data: [fgA * 100, fgNo * 100, fgB * 100],
                            backgroundColor: ['#1a56db', '#4b5563', '#e21a37'],
                            borderWidth: 0,
                            hoverOffset: 3
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        cutout: '70%',
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                position: 'followCursor',
                                callbacks: {
                                    label: function(context) {
                                        return ` ${context.label}: ${context.raw.toFixed(1)}%`;
                                    }
                                }
                            }
                        }
                    }
                });

                // -------------------------------------------------------------
                // 4. POPULATE KEY MATCH INSIGHTS CARDS
                // -------------------------------------------------------------
                // Card 1: Win Favored
                const insTitleFavored = document.getElementById('ins_title_favored');
                const insDescFavored = document.getElementById('ins_desc_favored');
                if (pWinA > pWinB) {
                    const isSlight = (pWinA - pWinB) <= 0.15;
                    insTitleFavored.innerText = `${teamA} is ${isSlight ? 'slightly favored' : 'favored'}`;
                    insDescFavored.innerText = `${(pWinA * 100).toFixed(1)}% win probability gives ${teamA} ${isSlight ? 'a slight' : 'a solid'} edge in this matchup.`;
                } else if (pWinB > pWinA) {
                    const isSlight = (pWinB - pWinA) <= 0.15;
                    insTitleFavored.innerText = `${teamB} is ${isSlight ? 'slightly favored' : 'favored'}`;
                    insDescFavored.innerText = `${(pWinB * 100).toFixed(1)}% win probability gives ${teamB} ${isSlight ? 'a slight' : 'a solid'} edge in this matchup.`;
                } else {
                    insTitleFavored.innerText = "Evenly Matchup";
                    insDescFavored.innerText = "Both teams have nearly identical win probabilities, projecting a highly competitive game.";
                }

                // Card 2: Goal Expectations
                const insTitleGoals = document.getElementById('ins_title_goals');
                const insDescGoals = document.getElementById('ins_desc_goals');
                const expTotalGoals = expGoalsA + expGoalsB;
                if (expTotalGoals >= 2.75) {
                    insTitleGoals.innerText = "High Scoring Match";
                    insDescGoals.innerText = `${expTotalGoals.toFixed(2)} average total goals suggests an open, attacking style of play from both sides.`;
                } else if (expTotalGoals < 2.0) {
                    insTitleGoals.innerText = "Low Scoring Match";
                    insDescGoals.innerText = `${expTotalGoals.toFixed(2)} average total goals suggests a tight, defensive battle with few scoring chances.`;
                } else {
                    insTitleGoals.innerText = "Standard Scoring";
                    insDescGoals.innerText = `${expTotalGoals.toFixed(2)} average total goals suggests a standard, competitive match with balanced action.`;
                }

                // Card 3: First Goal Advantage
                const insTitleFirstGoal = document.getElementById('ins_title_first_goal');
                const insDescFirstGoal = document.getElementById('ins_desc_first_goal');
                if (fgA >= fgB) {
                    insTitleFirstGoal.innerText = `${teamA} to score first`;
                    insDescFirstGoal.innerText = `${(fgA * 100).toFixed(1)}% probability of ${teamA} scoring the opening goal of the match.`;
                } else {
                    insTitleFirstGoal.innerText = `${teamB} to score first`;
                    insDescFirstGoal.innerText = `${(fgB * 100).toFixed(1)}% probability of ${teamB} scoring the opening goal of the match.`;
                }

                // Card 4: BTTS vs Clean Sheet
                const insTitleBtts = document.getElementById('ins_title_btts');
                const insDescBtts = document.getElementById('ins_desc_btts');
                if (bttsProb >= 0.50) {
                    insTitleBtts.innerText = "Both Teams to Score";
                    insDescBtts.innerText = `${(bttsProb * 100).toFixed(1)}% probability that both teams will find the back of the net.`;
                } else {
                    insTitleBtts.innerText = "Clean Sheet Likely";
                    insDescBtts.innerText = `${((1 - bttsProb) * 100).toFixed(1)}% probability that at least one team will fail to score.`;
                }

            } catch (err) {
                document.getElementById('spinner').style.display = 'none';
                document.getElementById('placeholder').style.display = 'flex';
                showAlert("Error simulating match. Please check console or models.");
                console.error(err);
            }
        });

        function showAlert(msg) {
            document.getElementById('custom_alert_message').innerText = msg;
            document.getElementById('custom_alert').style.display = 'flex';
        }

        function closeCustomAlert() {
            document.getElementById('custom_alert').style.display = 'none';
        }
    </script>

    <!-- Custom Alert Modal -->
    <div id="custom_alert" class="modal-overlay" style="display: none;">
        <div class="modal-card">
            <h3 class="modal-title">Attention</h3>
            <p id="custom_alert_message" class="modal-message">Please select two different teams.</p>
            <button class="modal-btn" onclick="closeCustomAlert()">OK</button>
        </div>
    </div>
</body>
</html>"""

@app.get("/docs", include_in_schema=False, response_class=HTMLResponse)
def custom_swagger_ui() -> str:
    """Returns a completely customized dark theme interactive API documentation dashboard matching the design mockup."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>World Cup Predictive Engine - API Docs</title>
    <link rel="icon" type="image/webp" href="/static/ball.webp">
    <link href="https://fonts.googleapis.com/css2?family=Anton&family=Oswald:wght@400;600;700;900&family=Noto+Sans:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #030408;
            --sidebar-bg: #060913;
            --card-bg: rgba(10, 12, 22, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-color: #ffffff;
            --text-muted: #9ca3af;
            
            /* Official FWC 26 Colors */
            --fwc-red: #e21a37;
            --fwc-blue: #1a56db;
            --fwc-cyan: #0ea5e9;
            --fwc-green: #16a34a;
            --fwc-yellow: #eab308;
            --fwc-purple: #9333ea;
            
            --primary: var(--fwc-blue);
            --primary-glow: rgba(26, 86, 219, 0.3);
            --accent: var(--fwc-red);
            --accent-glow: rgba(226, 26, 55, 0.3);
            
            --get-color: #0ea5e9;
            --get-bg: rgba(14, 165, 233, 0.15);
            --post-color: #10b981;
            --post-bg: rgba(16, 185, 129, 0.15);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Noto Sans', sans-serif;
            -ms-overflow-style: none;  /* IE and Edge */
            scrollbar-width: none;  /* Firefox */
        }

        *::-webkit-scrollbar {
            display: none; /* Chrome, Safari, Opera */
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            min-height: 100vh;
            overflow: hidden;
        }

        .app-layout {
            display: flex;
            min-height: 100vh;
        }

        /* Sidebar Styling */
        .sidebar {
            width: 280px;
            background: var(--sidebar-bg);
            border-right: 1px solid var(--border-color);
            padding: 2rem 1.2rem;
            display: flex;
            flex-direction: column;
            position: fixed;
            height: 100vh;
            box-sizing: border-box;
            z-index: 100;
        }

        .sidebar-logo-container {
            text-align: center;
            margin-bottom: 2rem;
        }

        .sidebar-logo {
            max-width: 140px;
            height: auto;
        }

        .sidebar-nav {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            flex-grow: 1;
            overflow-y: auto;
            padding-right: 4px;
        }

        .sidebar-section-title {
            font-family: 'Oswald', sans-serif;
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 1.5px;
            color: #4b5563;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            padding-left: 0.8rem;
        }

        .sidebar-nav-item {
            font-family: 'Oswald', sans-serif;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9rem;
            letter-spacing: 0.5px;
            color: #9ca3af;
            text-decoration: none;
            padding: 0.6rem 0.8rem;
            border-radius: 6px;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid transparent;
            cursor: pointer;
        }

        .sidebar-nav-item:hover {
            color: #ffffff;
            background: rgba(255, 255, 255, 0.03);
            border-color: rgba(255, 255, 255, 0.05);
        }

        .sidebar-nav-item.active {
            color: #ffffff !important;
            background: linear-gradient(135deg, rgba(26, 86, 219, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%) !important;
            border-color: rgba(139, 92, 246, 0.3) !important;
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.15);
        }

        .method-indicator {
            font-size: 0.7rem;
            font-weight: 900;
            padding: 1px 4px;
            border-radius: 4px;
            min-width: 32px;
            text-align: center;
        }

        .method-indicator.get {
            background: var(--get-bg);
            color: var(--get-color);
            border: 1px solid rgba(14, 165, 233, 0.3);
        }

        .method-indicator.post {
            background: var(--post-bg);
            color: var(--post-color);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .sidebar-status-card {
            background: rgba(10, 12, 22, 0.5);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.8rem;
            margin-top: 1rem;
            font-size: 0.8rem;
        }

        .status-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 700;
            color: #10b981;
            margin-bottom: 0.5rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px #10b981;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        .status-row {
            display: flex;
            justify-content: space-between;
            margin-top: 0.3rem;
            color: var(--text-muted);
        }

        .status-row strong {
            color: #ffffff;
        }

        .sidebar-footer {
            border-top: 1px solid var(--border-color);
            padding-top: 1rem;
            margin-top: 1rem;
        }

        .sidebar-author {
            font-family: 'Oswald', sans-serif;
            font-size: 0.9rem;
            font-weight: 700;
            color: #ffffff;
        }

        .sidebar-author-title {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .sidebar-copyright {
            font-size: 0.65rem;
            color: #4b5563;
            margin-top: 0.4rem;
        }

        /* Main Content Layout */
        .main-content {
            margin-left: 280px;
            flex-grow: 1;
            display: grid;
            grid-template-columns: 1fr 480px;
            min-height: 100vh;
        }

        /* Middle Column: Endpoints List */
        .endpoints-column {
            padding: 3rem 2.5rem;
            border-right: 1px solid var(--border-color);
            background: radial-gradient(circle at top left, rgba(26, 86, 219, 0.03), transparent 50%);
            height: 100vh;
            overflow-y: auto;
        }

        .api-header {
            margin-bottom: 3rem;
        }

        .api-header h1 {
            font-family: 'Anton', sans-serif;
            font-size: 2.8rem;
            letter-spacing: 0.5px;
            line-height: 1.1;
            background: linear-gradient(135deg, #ffffff 0%, #a1a1aa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .api-header p {
            color: var(--text-muted);
            font-size: 1.05rem;
            line-height: 1.5;
            max-width: 700px;
        }

        .endpoints-section h2 {
            font-family: 'Oswald', sans-serif;
            font-size: 1.4rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
        }

        .endpoint-card {
            background: rgba(10, 12, 22, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            cursor: pointer;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .endpoint-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            width: 3px;
            background: transparent;
            transition: all 0.2s;
        }

        .endpoint-card:hover {
            background: rgba(10, 12, 22, 0.65);
            border-color: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        }

        .endpoint-card.active {
            background: rgba(10, 12, 22, 0.85);
            border-color: rgba(26, 86, 219, 0.4);
            box-shadow: 0 4px 20px rgba(26, 86, 219, 0.15);
        }

        .endpoint-card.active::before {
            background: var(--fwc-blue);
        }

        .card-header {
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }

        .badge {
            font-family: 'Oswald', sans-serif;
            font-weight: 700;
            font-size: 0.8rem;
            padding: 3px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }

        .badge.get {
            background: var(--get-bg);
            color: var(--get-color);
            border: 1px solid rgba(14, 165, 233, 0.3);
        }

        .badge.post {
            background: var(--post-bg);
            color: var(--post-color);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .path {
            font-family: monospace;
            font-size: 1.05rem;
            font-weight: 600;
            color: #ffffff;
        }

        .summary {
            font-family: 'Oswald', sans-serif;
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-left: 0.5rem;
        }

        .status-badge {
            margin-left: auto;
            font-family: 'Oswald', sans-serif;
            font-size: 0.8rem;
            font-weight: 700;
            color: #10b981;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            padding: 2px 6px;
            border-radius: 4px;
        }

        .chevron {
            color: var(--text-muted);
            transition: transform 0.2s;
            margin-left: 0.5rem;
        }

        .endpoint-card:hover .chevron {
            color: #ffffff;
        }

        .card-description {
            margin-top: 0.8rem;
            color: var(--text-muted);
            font-size: 0.9rem;
            line-height: 1.4;
        }

        /* Right Column: API Explorer Panel */
        .explorer-column {
            background: #05070e;
            border-left: 1px solid var(--border-color);
            padding: 2.5rem 2rem;
            height: 100vh;
            overflow-y: auto;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .explorer-top-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.2rem;
        }

        .api-version-container {
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
        }

        .api-version-container span:first-child {
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 700;
        }

        .version-badge {
            font-family: 'Oswald', sans-serif;
            font-weight: 700;
            background: var(--fwc-purple);
            color: #ffffff;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            width: fit-content;
            box-shadow: 0 4px 10px rgba(147, 51, 234, 0.3);
        }

        .btn-download-openapi {
            font-family: 'Oswald', sans-serif;
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.8rem;
            text-decoration: none;
            color: #ffffff;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            padding: 0.5rem 0.8rem;
            border-radius: 6px;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s;
        }

        .btn-download-openapi:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .explorer-panel {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .explorer-header {
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }

        .explorer-path {
            font-family: monospace;
            font-size: 1.15rem;
            font-weight: 700;
            color: #ffffff;
        }

        .explorer-desc {
            color: var(--text-muted);
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .explorer-section {
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
        }

        .explorer-section h3 {
            font-family: 'Oswald', sans-serif;
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #ffffff;
            border-left: 3px solid var(--fwc-cyan);
            padding-left: 0.5rem;
        }

        /* Parameters input table/list */
        .params-list {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .param-row {
            display: grid;
            grid-template-columns: 1fr 140px;
            gap: 1rem;
            align-items: center;
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid var(--border-color);
            padding: 0.8rem;
            border-radius: 6px;
        }

        .param-info {
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
        }

        .param-name {
            font-family: monospace;
            font-weight: 700;
            color: var(--fwc-cyan);
            font-size: 0.85rem;
        }

        .param-type {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .param-type .required {
            color: var(--fwc-red);
            font-weight: 700;
            margin-left: 4px;
        }

        .param-desc {
            font-size: 0.75rem;
            color: #9ca3af;
            line-height: 1.3;
            margin-top: 0.2rem;
        }

        .explorer-input {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 0.4rem 0.6rem;
            color: #ffffff;
            font-size: 0.85rem;
            outline: none;
            width: 100%;
            transition: all 0.2s;
        }

        .explorer-input:focus {
            border-color: var(--fwc-blue);
            background: rgba(255, 255, 255, 0.08);
        }

        /* Code display block */
        .code-wrapper {
            position: relative;
            background: #090d22;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 1rem;
        }

        .code-wrapper pre {
            margin: 0;
            overflow-x: auto;
        }

        .code-wrapper code {
            font-family: monospace;
            font-size: 0.85rem;
            line-height: 1.4;
            display: block;
            white-space: pre;
        }

        .btn-copy {
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 2px 6px;
            color: var(--text-muted);
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-copy:hover {
            color: #ffffff;
            background: rgba(255, 255, 255, 0.1);
        }

        /* Explorer action button */
        .btn-try-explorer {
            background: linear-gradient(135deg, var(--fwc-red) 0%, #b0152b 100%);
            color: #ffffff;
            border: none;
            border-radius: 6px;
            padding: 0.8rem;
            font-family: 'Oswald', sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(226, 26, 55, 0.35);
            text-transform: uppercase;
            letter-spacing: 1px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.6rem;
            width: 100%;
        }

        .btn-try-explorer:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 25px rgba(226, 26, 55, 0.5);
            background: linear-gradient(135deg, #f43f5e 0%, var(--fwc-red) 100%);
        }

        .btn-try-explorer:active {
            transform: translateY(0);
        }

        /* JSON syntax highlight classes */
        .json-key { color: #38bdf8; font-weight: 600; }
        .json-string { color: #a3e635; }
        .json-number { color: #fb923c; }
        .json-boolean { color: #c084fc; }
        .json-null { color: #9ca3af; }

        .explorer-section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .response-status-label {
            font-family: 'Oswald', sans-serif;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 1px 6px;
            border-radius: 4px;
        }

        .response-status-label.status-pending { background: rgba(234, 179, 8, 0.1); color: var(--fwc-yellow); border: 1px solid rgba(234, 179, 8, 0.2); }
        .response-status-label.status-success { background: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.2); }
        .response-status-label.status-error { background: rgba(239, 68, 68, 0.1); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.2); }

        /* Mobile Header */
        .mobile-header {
            display: none;
            background: var(--sidebar-bg);
            border-bottom: 1px solid var(--border-color);
            padding: 0.8rem 1.2rem;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 200;
        }

        .mobile-logo {
            max-height: 32px;
            width: auto;
        }

        .hamburger-btn {
            background: transparent;
            border: none;
            color: #ffffff;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0.2rem;
        }
        
        .sidebar-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(3, 4, 8, 0.6);
            backdrop-filter: blur(4px);
            z-index: 150;
        }
        
        .sidebar-overlay.active {
            display: block;
        }

        /* Responsive */
        @media (max-width: 1024px) {
            body {
                overflow: auto !important;
            }
            
            .mobile-header {
                display: flex !important;
            }

            .sidebar {
                display: flex !important;
                position: fixed;
                top: 0;
                left: 0;
                width: 280px;
                height: 100vh;
                z-index: 160;
                transform: translateX(-100%);
                transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                border-right: 1px solid var(--border-color);
            }

            .sidebar.mobile-open {
                transform: translateX(0);
                box-shadow: 5px 0 25px rgba(0, 0, 0, 0.8);
            }

            .main-content {
                margin-left: 0;
                grid-template-columns: 1fr;
                min-width: 0 !important;
            }

            .endpoints-column {
                height: auto !important;
                overflow-y: visible !important;
                padding: 2rem 1.25rem !important;
                min-width: 0 !important;
            }

            .explorer-column {
                height: auto !important;
                overflow-y: visible !important;
                position: static;
                border-left: none;
                border-top: 1px solid var(--border-color);
                padding: 2rem 1.25rem !important;
                min-width: 0 !important;
            }
        }

        @media (max-width: 600px) {
            .api-header h1 {
                font-size: 1.8rem !important;
            }
            .card-header {
                flex-wrap: wrap !important;
                gap: 0.5rem 0.8rem !important;
            }
            .status-badge {
                margin-left: 0 !important;
            }
            .chevron {
                margin-left: auto !important;
            }
            .path {
                font-size: 0.9rem !important;
                word-break: break-all !important;
            }
            .summary {
                font-size: 0.85rem !important;
                width: 100% !important;
                margin-left: 0 !important;
            }
            .param-row {
                grid-template-columns: 1fr !important;
                gap: 0.8rem !important;
            }
            .param-input-container {
                width: 100% !important;
            }
            .code-wrapper {
                max-width: 100% !important;
                box-sizing: border-box !important;
            }
        }
    </style>
</head>
<body>
    <!-- Mobile Header -->
    <header class="mobile-header">
        <button class="hamburger-btn" onclick="toggleMobileSidebar(event)">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="3" y1="12" x2="21" y2="12"></line>
                <line x1="3" y1="6" x2="21" y2="6"></line>
                <line x1="3" y1="18" x2="21" y2="18"></line>
            </svg>
        </button>
        <img src="/static/logo.png" alt="World Cup 2026 Logo" class="mobile-logo">
        <div style="width: 24px;"></div>
    </header>

    <!-- Sidebar Overlay -->
    <div class="sidebar-overlay" onclick="toggleMobileSidebar(event)"></div>

    <div class="app-layout">
        <!-- Sidebar -->
        <aside class="sidebar">
            <!-- Logo -->
            <div class="sidebar-logo-container">
                <img src="/static/logo.png" alt="World Cup 2026 Logo" class="sidebar-logo">
            </div>
            <!-- Main Nav -->
            <nav class="sidebar-nav">
                <div class="sidebar-section-title">Navigation</div>
                <a class="sidebar-nav-item active" id="nav-overview" onclick="showOverview()">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
                    Overview
                </a>
                <a class="sidebar-nav-item" id="nav-auth" onclick="showAuth()">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
                    Authentication
                </a>
                
                <div class="sidebar-section-title">Endpoints</div>
                <a class="sidebar-nav-item" id="nav-health" onclick="selectEndpoint('health')">
                    <span class="method-indicator get">GET</span> Health
                </a>
                <a class="sidebar-nav-item" id="nav-teams" onclick="selectEndpoint('teams')">
                    <span class="method-indicator get">GET</span> Teams
                </a>
                <a class="sidebar-nav-item" id="nav-referees" onclick="selectEndpoint('referees')">
                    <span class="method-indicator get">GET</span> Referees
                </a>
                <a class="sidebar-nav-item" id="nav-predict" onclick="selectEndpoint('predict')">
                    <span class="method-indicator get">GET</span> Predict
                </a>
                <div class="sidebar-section-title">Models</div>
                <a class="sidebar-nav-item" id="nav-scrape" onclick="selectEndpoint('scrape')">
                    <span class="method-indicator post">POST</span> Scrape
                </a>
                <a class="sidebar-nav-item" id="nav-train" onclick="selectEndpoint('train')">
                    <span class="method-indicator post">POST</span> Train
                </a>
                
                <div class="sidebar-section-title">Other Links</div>
                <a href="/" class="sidebar-nav-item" style="color: var(--fwc-cyan); border-color: rgba(14, 165, 233, 0.3); background: rgba(14, 165, 233, 0.1);">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path></svg>
                    Simulation UI
                </a>
                <a href="https://github.com/SebasHuaypar" target="_blank" class="sidebar-nav-item">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>
                    GitHub
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-left: auto;"><line x1="7" y1="17" x2="17" y2="7"></line><polyline points="7 7 17 7 17 17"></polyline></svg>
                </a>
            </nav>
            
            <!-- Author Footer -->
            <div class="sidebar-footer">
                <div class="sidebar-author">Sebastián Huaypar Acurio</div>
                <div class="sidebar-author-title">Computer Science @ UNI</div>
            </div>
        </aside>
        
        <!-- Main Content Area -->
        <div class="main-content">
            <!-- Middle Column: Endpoints List -->
            <div class="endpoints-column">
                <div class="api-header">
                    <h1>WORLD CUP PREDICTIVE ENGINE API</h1>
                    <p>REST API to predict World Cup match outcomes (goals, corners, cards, knockouts) using Machine Learning models and Monte Carlo simulations.</p>
                </div>
                
                <div class="endpoints-section">
                    <h2>API Endpoints</h2>
                    
                    <!-- Card: Health -->
                    <div class="endpoint-card" id="card-health" onclick="selectEndpoint('health')">
                        <div class="card-header">
                            <span class="badge get">GET</span>
                            <code class="path">/api/v1/health</code>
                            <span class="summary">Health Check</span>
                            <span class="status-badge">200 OK</span>
                            <svg class="chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
                        </div>
                        <div class="card-description">Returns the operational status of the predictive engine.</div>
                    </div>
                    
                    <!-- Card: Teams -->
                    <div class="endpoint-card" id="card-teams" onclick="selectEndpoint('teams')">
                        <div class="card-header">
                            <span class="badge get">GET</span>
                            <code class="path">/api/v1/teams</code>
                            <span class="summary">Get Teams</span>
                            <span class="status-badge">200 OK</span>
                            <svg class="chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
                        </div>
                        <div class="card-description">Loads all distinct national teams currently present in the historical match database.</div>
                    </div>

                    <!-- Card: Referees -->
                    <div class="endpoint-card" id="card-referees" onclick="selectEndpoint('referees')">
                        <div class="card-header">
                            <span class="badge get">GET</span>
                            <code class="path">/api/v1/referees</code>
                            <span class="summary">Get Referees</span>
                            <span class="status-badge">200 OK</span>
                            <svg class="chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
                        </div>
                        <div class="card-description">Loads the full catalog of match referees together with their average card statistics.</div>
                    </div>
                    
                    <!-- Card: Predict -->
                    <div class="endpoint-card" id="card-predict" onclick="selectEndpoint('predict')">
                        <div class="card-header">
                            <span class="badge get">GET</span>
                            <code class="path">/api/v1/predict</code>
                            <span class="summary">Predict Match Outcome</span>
                            <span class="status-badge">200 OK</span>
                            <svg class="chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
                        </div>
                        <div class="card-description">Performs the full Monte Carlo simulation matching ELO, market values, absences, altitude, and referee ratings.</div>
                    </div>
                    
                    <!-- Card: Scrape -->
                    <div class="endpoint-card" id="card-scrape" onclick="selectEndpoint('scrape')">
                        <div class="card-header">
                            <span class="badge post">POST</span>
                            <code class="path">/api/v1/admin/scrape</code>
                            <span class="summary">Trigger Scraping</span>
                            <span class="status-badge">200 OK</span>
                            <svg class="chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
                        </div>
                        <div class="card-description">Runs the scraper background task to ingest matches, injuries, and ratings. Protected by API key.</div>
                    </div>
                    
                    <!-- Card: Train -->
                    <div class="endpoint-card" id="card-train" onclick="selectEndpoint('train')">
                        <div class="card-header">
                            <span class="badge post">POST</span>
                            <code class="path">/api/v1/admin/train</code>
                            <span class="summary">Trigger Model Training</span>
                            <span class="status-badge">200 OK</span>
                            <svg class="chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
                        </div>
                        <div class="card-description">Triggers XGBoost model retraining using updated SQLite features. Protected by API key.</div>
                    </div>
                </div>
            </div>
            
            <!-- Right Column: API Explorer Panel -->
            <div class="explorer-column">
                <div class="explorer-top-bar">
                    <div class="api-version-container">
                        <span>API Version</span>
                        <span class="version-badge">v1.0.0</span>
                    </div>
                    <a href="/openapi.json" class="btn-download-openapi" download>
                        Download OpenAPI
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                    </a>
                </div>
                
                <div class="explorer-panel" id="explorer-panel">
                    <!-- Dynamic content injected here -->
                </div>
            </div>
        </div>
    </div>

    <script>
        function toggleMobileSidebar(event) {
            if (event) event.stopPropagation();
            const sidebar = document.querySelector('.sidebar');
            const overlay = document.querySelector('.sidebar-overlay');
            if (sidebar) sidebar.classList.toggle('mobile-open');
            if (overlay) overlay.classList.toggle('active');
        }

        function closeMobileSidebar() {
            const sidebar = document.querySelector('.sidebar');
            const overlay = document.querySelector('.sidebar-overlay');
            if (sidebar) sidebar.classList.remove('mobile-open');
            if (overlay) overlay.classList.remove('active');
        }

        const ENDPOINTS_DATA = {
            health: {
                name: "health",
                method: "GET",
                path: "/api/v1/health",
                summary: "Health Check",
                description: "Returns the health status of the API. Useful for Uptime monitors.",
                parameters: [],
                responseExample: {
                    "status": "ok",
                    "message": "Predictive Engine API is running smoothly."
                },
                curlExample: `curl -X GET "http://localhost:8000/api/v1/health"`
            },
            teams: {
                name: "teams",
                method: "GET",
                path: "/api/v1/teams",
                summary: "Get Teams",
                description: "Returns a list of all national teams available in the database.",
                parameters: [],
                responseExample: {
                    "count": 52,
                    "teams": ["Argentina", "Brazil", "France", "Germany", "Spain", "Italy", "Uruguay"]
                },
                curlExample: `curl -X GET "http://localhost:8000/api/v1/teams"`
            },
            referees: {
                name: "referees",
                method: "GET",
                path: "/api/v1/referees",
                summary: "Get Referees",
                description: "Returns a list of all referees available in the database with their stats.",
                parameters: [],
                responseExample: {
                    "count": 52,
                    "referees": [
                        {
                            "referee": "Jesús Valenzuela",
                            "matches_played": 30,
                            "yellow_cards_per_match": 5.1,
                            "red_cards_per_match": 0.25,
                            "fouls_per_match": 29.5
                        },
                        {
                            "referee": "Facundo Tello",
                            "matches_played": 32,
                            "yellow_cards_per_match": 4.8,
                            "red_cards_per_match": 0.30,
                            "fouls_per_match": 27.5
                        }
                    ]
                },
                curlExample: `curl -X GET "http://localhost:8000/api/v1/referees"`
            },
            predict: {
                name: "predict",
                method: "GET",
                path: "/api/v1/predict",
                summary: "Predict Match Outcome",
                description: "Runs a Monte Carlo simulation of a match between Team A and Team B and returns detailed statistics.",
                parameters: [
                    { name: "team_a", type: "string", required: true, default: "Argentina", description: "Name of National Team A (e.g. Argentina)" },
                    { name: "team_b", type: "string", required: true, default: "France", description: "Name of National Team B (e.g. France)" },
                    { name: "neutral", type: "boolean", required: false, default: "true", description: "Is the match played on neutral ground? (Default: True)" },
                    { name: "n_sims", type: "integer", required: false, default: "10000", description: "Number of Monte Carlo simulations (1,000 to 100,000)" },
                    { name: "missing_players_a", type: "integer", required: false, default: "0", description: "Number of key players absent for Team A" },
                    { name: "missing_players_b", type: "integer", required: false, default: "0", description: "Number of key players absent for Team B" },
                    { name: "knockout", type: "boolean", required: false, default: "false", description: "Enable extra time and penalties on draw" },
                    { name: "city", type: "string", required: false, default: "", description: "City hosting the match (altitude check)" },
                    { name: "country", type: "string", required: false, default: "", description: "Country hosting the match" },
                    { name: "referee", type: "string", required: false, default: "", description: "Name of match referee (scales cards)" }
                ],
                responseExample: {
                    "match_info": {
                        "team_a": "Argentina",
                        "team_b": "France",
                        "elo_a_raw": 2178.45,
                        "elo_b_raw": 2110.15,
                        "elo_a_adjusted": 2178.45,
                        "elo_b_adjusted": 2110.15,
                        "neutral": true,
                        "missing_players_a": 0,
                        "missing_players_b": 0,
                        "knockout": false,
                        "city": null,
                        "country": null,
                        "altitude_meters": 0.0,
                        "referee": null,
                        "squad_value_a": 780000000.0,
                        "squad_value_b": 950000000.0
                    },
                    "predictions": {
                        "outcomes": {
                            "win_a": 0.3805,
                            "win_b": 0.3436,
                            "draw": 0.2759
                        },
                        "goals": {
                            "expected_a": 1.21,
                            "expected_b": 1.14,
                            "expected_total": 2.35
                        }
                    }
                },
                curlExample: `curl -X GET "http://localhost:8000/api/v1/predict?team_a=Argentina&team_b=France&neutral=true&n_sims=10000"`
            },
            scrape: {
                name: "scrape",
                method: "POST",
                path: "/api/v1/admin/scrape",
                summary: "Trigger Scraping",
                description: "Triggers the full scraping and ELO calculation pipeline in the background. Requires administrative API Key.",
                headers: [
                    { name: "X-Admin-API-Key", type: "string", required: true, default: "", description: "Administrative API secret key" }
                ],
                parameters: [],
                responseExample: {
                    "message": "Scraping pipeline started in background."
                },
                curlExample: `curl -X POST "http://localhost:8000/api/v1/admin/scrape" \\\n  -H "X-Admin-API-Key: YOUR_ADMIN_SECRET"`
            },
            train: {
                name: "train",
                method: "POST",
                path: "/api/v1/admin/train",
                summary: "Trigger Model Training",
                description: "Triggers the ML model retraining pipeline in the background. Requires administrative API Key.",
                headers: [
                    { name: "X-Admin-API-Key", type: "string", required: true, default: "", description: "Administrative API secret key" }
                ],
                parameters: [],
                responseExample: {
                    "message": "Model retraining started in background."
                },
                curlExample: `curl -X POST "http://localhost:8000/api/v1/admin/train" \\\n  -H "X-Admin-API-Key: YOUR_ADMIN_SECRET"`
            }
        };

        const OVERVIEW_HTML = `
            <div class="explorer-header">
                <h2 style="font-family: 'Oswald', sans-serif; font-size: 1.6rem; text-transform: uppercase;">Overview</h2>
            </div>
            <div style="line-height: 1.6; color: var(--text-muted); display: flex; flex-direction: column; gap: 1rem;">
                <p>Welcome to the <strong>World Cup Predictive Engine API</strong> documentation.</p>
                <p>This predictive engine uses an XGBoost regression model trained on historical match results, squad market values, and advanced tactical metrics to estimate expected goals for individual matches.</p>
                <p>Monte Carlo simulations are run dynamically on top of the goals model to compute exact match probabilities, expected corners, cards, and knockout-stage progressions.</p>
                <div class="explorer-section" style="margin-top: 1rem;">
                    <h3>Base URL</h3>
                    <code style="background: #090d22; padding: 0.6rem 0.8rem; display: block; border-radius: 6px; border: 1px solid var(--border-color); color: var(--fwc-cyan); font-family: monospace; font-size: 0.9rem;">
                        /api/v1
                    </code>
                </div>
            </div>
        `;

        const AUTH_HTML = `
            <div class="explorer-header">
                <h2 style="font-family: 'Oswald', sans-serif; font-size: 1.6rem; text-transform: uppercase;">Authentication</h2>
            </div>
            <div style="line-height: 1.6; color: var(--text-muted); display: flex; flex-direction: column; gap: 1rem;">
                <p>Most endpoints of the predictive engine are public and require no authentication.</p>
                <p>However, administrative endpoints (such as trigger scraping and trigger model training) are protected to prevent denial of service and unauthorized operations.</p>
                <div class="explorer-section" style="margin-top: 1rem;">
                    <h3>Header Authentication</h3>
                    <p>To access protected endpoints, you must send your secret API key in the request header:</p>
                    <div style="background: #090d22; padding: 0.8rem; border-radius: 6px; border: 1px solid var(--border-color); font-family: monospace; color: #e5e7eb; font-size: 0.85rem; margin-top: 0.5rem;">
                        X-Admin-API-Key: YOUR_ADMIN_SECRET
                    </div>
                </div>
                <p style="font-size: 0.85rem;">You can configure your secret key via the <code>ADMIN_API_KEY</code> environment variable on your server deployment.</p>
            </div>
        `;

        function highlightJson(json) {
            if (typeof json !== 'string') {
                json = JSON.stringify(json, null, 2);
            }
            json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g, function (match) {
                var cls = 'number';
                if (/^"/.test(match)) {
                    if (/:$/.test(match)) {
                        cls = 'key';
                    } else {
                        cls = 'string';
                    }
                } else if (/true|false/.test(match)) {
                    cls = 'boolean';
                } else if (/null/.test(match)) {
                    cls = 'null';
                }
                return '<span class="json-' + cls + '">' + match + '</span>';
            });
        }

        function copyText(elementId) {
            const text = document.getElementById(elementId).innerText;
            navigator.clipboard.writeText(text).then(() => {
                const btn = document.querySelector(`#${elementId}`).parentElement.querySelector('.btn-copy');
                if (btn) {
                    const oldText = btn.innerText;
                    btn.innerText = "Copied!";
                    setTimeout(() => { btn.innerText = oldText; }, 2000);
                }
            });
        }

        function showOverview() {
            closeMobileSidebar();
            deactivateAll();
            document.getElementById('nav-overview').classList.add('active');
            document.getElementById('explorer-panel').innerHTML = OVERVIEW_HTML;
        }

        function showAuth() {
            closeMobileSidebar();
            deactivateAll();
            document.getElementById('nav-auth').classList.add('active');
            document.getElementById('explorer-panel').innerHTML = AUTH_HTML;
        }

        function deactivateAll() {
            // Sidebar
            const navItems = document.querySelectorAll('.sidebar-nav-item');
            navItems.forEach(item => item.classList.remove('active'));
            // Cards
            const cards = document.querySelectorAll('.endpoint-card');
            cards.forEach(card => card.classList.remove('active'));
        }

        function selectEndpoint(id) {
            closeMobileSidebar();
            deactivateAll();
            
            // Activate sidebar item and card
            const navItem = document.getElementById(`nav-${id}`);
            if (navItem) navItem.classList.add('active');
            
            const card = document.getElementById(`card-${id}`);
            if (card) {
                card.classList.add('active');
                card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
            
            const endpoint = ENDPOINTS_DATA[id];
            
            let html = `
                <div class="explorer-header">
                    <span class="badge ${endpoint.method.toLowerCase()}">${endpoint.method}</span>
                    <code class="explorer-path">${endpoint.path}</code>
                </div>
                <p class="explorer-desc">${endpoint.description}</p>
            `;
            
            html += renderHeadersSection(endpoint);
            html += renderParametersSection(endpoint);
            
            html += `
                <div class="explorer-section">
                    <h3>Example Request</h3>
                    <div class="code-wrapper">
                        <pre><code id="explorer-curl">${endpoint.curlExample}</code></pre>
                        <button class="btn-copy" onclick="copyText('explorer-curl')">Copy</button>
                    </div>
                </div>

                <div class="explorer-section">
                    <div class="explorer-section-header">
                        <h3>Response</h3>
                        <span class="response-status-label" id="response-status">Example</span>
                    </div>
                    <div class="code-wrapper">
                        <pre><code id="explorer-response">${highlightJson(endpoint.responseExample)}</code></pre>
                        <button class="btn-copy" onclick="copyText('explorer-response')">Copy</button>
                    </div>
                </div>

                <div class="explorer-actions">
                    <button class="btn-try-explorer" onclick="executeApiCall('${id}')">
                        Try in API Explorer
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                    </button>
                </div>
            `;
            
            document.getElementById('explorer-panel').innerHTML = html;
        }

        function renderHeadersSection(endpoint) {
            if (!endpoint.headers || endpoint.headers.length === 0) return '';
            let html = `
                <div class="explorer-section">
                    <h3>Headers</h3>
                    <div class="params-list">
            `;
            endpoint.headers.forEach(h => {
                html += `
                    <div class="param-row">
                        <div class="param-info">
                            <span class="param-name">${h.name}</span>
                            <span class="param-type">${h.type} <span class="required">*required</span></span>
                            <p class="param-desc">${h.description}</p>
                        </div>
                        <div class="param-input-container">
                            <input type="text" id="header-${h.name}" class="explorer-input" placeholder="Value..." oninput="updateCurl('${endpoint.name}')" />
                        </div>
                    </div>
                `;
            });
            html += `</div></div>`;
            return html;
        }

        function renderParametersSection(endpoint) {
            if (!endpoint.parameters || endpoint.parameters.length === 0) return '';
            let html = `
                <div class="explorer-section">
                    <h3>Query Parameters</h3>
                    <div class="params-list">
            `;
            endpoint.parameters.forEach(p => {
                let inputHtml = '';
                if (p.type === 'boolean') {
                    inputHtml = `
                        <select id="param-${p.name}" class="explorer-input" onchange="updateCurl('${endpoint.name}')">
                            <option value="true" ${p.default === 'true' ? 'selected' : ''}>true</option>
                            <option value="false" ${p.default === 'false' ? 'selected' : ''}>false</option>
                        </select>
                    `;
                } else {
                    inputHtml = `
                        <input type="${p.type === 'integer' ? 'number' : 'text'}" 
                               id="param-${p.name}" 
                               class="explorer-input" 
                               value="${p.default}" 
                               placeholder="${p.name}" 
                               oninput="updateCurl('${endpoint.name}')" />
                    `;
                }
                
                html += `
                    <div class="param-row">
                        <div class="param-info">
                            <span class="param-name">${p.name}</span>
                            <span class="param-type">${p.type} ${p.required ? '<span class="required">*required</span>' : ''}</span>
                            <p class="param-desc">${p.description}</p>
                        </div>
                        <div class="param-input-container">
                            ${inputHtml}
                        </div>
                    </div>
                `;
            });
            html += `</div></div>`;
            return html;
        }

        function updateCurl(id) {
            const endpoint = ENDPOINTS_DATA[id];
            const baseUrl = window.location.origin;
            let url = `${baseUrl}${endpoint.path}`;
            let queryParams = [];
            
            if (endpoint.parameters) {
                endpoint.parameters.forEach(p => {
                    const el = document.getElementById(`param-${p.name}`);
                    if (el) {
                        const val = el.value.trim();
                        if (val !== "") {
                            queryParams.push(`${p.name}=${encodeURIComponent(val)}`);
                        }
                    }
                });
            }
            
            if (queryParams.length > 0) {
                url += `?${queryParams.join('&')}`;
            }
            
            let curl = "";
            if (endpoint.method === "GET") {
                curl = `curl -X GET "${url}"`;
            } else {
                curl = `curl -X POST "${url}"`;
                if (endpoint.headers) {
                    endpoint.headers.forEach(h => {
                        const el = document.getElementById(`header-${h.name}`);
                        const val = el ? el.value.trim() : "";
                        curl += ` \\\n  -H "${h.name}: ${val || 'YOUR_ADMIN_SECRET'}"`;
                    });
                }
            }
            
            const curlEl = document.getElementById('explorer-curl');
            if (curlEl) {
                curlEl.innerText = curl;
            }
        }

        async function executeApiCall(id) {
            const endpoint = ENDPOINTS_DATA[id];
            let url = endpoint.path;
            let queryParams = [];
            
            if (endpoint.parameters) {
                endpoint.parameters.forEach(p => {
                    const el = document.getElementById(`param-${p.name}`);
                    if (el) {
                        const val = el.value.trim();
                        if (val !== "") {
                            queryParams.push(`${p.name}=${encodeURIComponent(val)}`);
                        }
                    }
                });
            }
            
            if (queryParams.length > 0) {
                url += `?${queryParams.join('&')}`;
            }
            
            const options = {
                method: endpoint.method
            };
            
            if (endpoint.headers) {
                options.headers = {};
                endpoint.headers.forEach(h => {
                    const el = document.getElementById(`header-${h.name}`);
                    if (el && el.value.trim() !== "") {
                        options.headers[h.name] = el.value.trim();
                    }
                });
            }
            
            const statusLabel = document.getElementById('response-status');
            const responseEl = document.getElementById('explorer-response');
            
            statusLabel.innerText = "Sending...";
            statusLabel.className = "response-status-label status-pending";
            
            try {
                const res = await fetch(url, options);
                const data = await res.json();
                
                statusLabel.innerText = `${res.status} ${res.statusText || (res.status === 200 ? 'OK' : '')}`;
                if (res.status >= 200 && res.status < 300) {
                    statusLabel.className = "response-status-label status-success";
                } else {
                    statusLabel.className = "response-status-label status-error";
                }
                
                responseEl.innerHTML = highlightJson(JSON.stringify(data, null, 2));
            } catch (err) {
                statusLabel.innerText = "Error";
                statusLabel.className = "response-status-label status-error";
                responseEl.innerHTML = highlightJson(JSON.stringify({ error: err.message }, null, 2));
            }
        }

        // Initialize view on page load
        window.addEventListener('DOMContentLoaded', () => {
            showOverview();
        });
    </script>
</body>
</html>"""


@app.get("/api/v1/health", tags=["General"])
def health_check() -> Dict[str, str]:
    """Returns the health status of the API."""
    return {"status": "ok", "message": "Predictive Engine API is running smoothly."}

@app.get("/api/v1/teams", tags=["Data"])
def get_teams() -> Dict[str, Any]:
    """Returns a list of all national teams available in the database."""
    try:
        with closing(get_connection()) as conn:
            cursor = conn.cursor()
            # Get unique teams from historical matches
            cursor.execute("""
                SELECT DISTINCT team FROM (
                    SELECT home_team AS team FROM matches
                    UNION
                    SELECT away_team AS team FROM matches
                ) 
                WHERE team IS NOT NULL AND team != ''
                ORDER BY team ASC
            """)
            teams = [row['team'] for row in cursor.fetchall()]
        return {"count": len(teams), "teams": teams}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/v1/referees", tags=["Data"])
def get_referees() -> Dict[str, Any]:
    """Returns a list of all referees available in the database with their stats."""
    try:
        referees = load_all_referees()
        return {"count": len(referees), "referees": referees}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/v1/predict", tags=["Prediction"])
def predict_match(
    team_a: str = Query(..., description="Name of National Team A (e.g., Argentina)"),
    team_b: str = Query(..., description="Name of National Team B (e.g., France)"),
    neutral: bool = Query(True, description="Is the match played on neutral ground? (Default: True for World Cup matches)"),
    n_sims: int = Query(10000, ge=1000, le=100000, description="Number of Monte Carlo simulations to execute"),
    missing_players_a: int = Query(0, ge=0, description="Number of key players absent for Team A"),
    missing_players_b: int = Query(0, ge=0, description="Number of key players absent for Team B"),
    knockout: bool = Query(False, description="Is it a knockout stage match? (Enables extra time and penalty shootout on draw)"),
    city: str = Query(None, description="City hosting the match (e.g., Mexico City)"),
    country: str = Query(None, description="Country hosting the match (e.g., Mexico)"),
    referee: str = Query(None, description="Name of the designated match referee (e.g., Szymon Marciniak)")
) -> Dict[str, Any]:
    """Runs a Monte Carlo simulation of a match between Team A and Team B and returns detailed statistics."""
    # Validate that team A and team B are not the same team
    if team_a.lower().strip() == team_b.lower().strip():
        raise HTTPException(
            status_code=400,
            detail="Team A and Team B cannot be the same national team."
        )
        
    # Validate that teams exist in the database
    with closing(get_connection()) as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT team FROM team_elo WHERE team = ? COLLATE NOCASE LIMIT 1", (team_a,))
        row_a = cursor.fetchone()
        
        cursor.execute("SELECT team FROM team_elo WHERE team = ? COLLATE NOCASE LIMIT 1", (team_b,))
        row_b = cursor.fetchone()
    
    if not row_a:
        raise HTTPException(
            status_code=400, 
            detail=f"Team '{team_a}' not found in the database. Please check the list of valid teams at /api/v1/teams."
        )
    if not row_b:
        raise HTTPException(
            status_code=400, 
            detail=f"Team '{team_b}' not found in the database. Please check the list of valid teams at /api/v1/teams."
        )
        
    # Use standard casing from database
    team_a = row_a['team']
    team_b = row_b['team']
        
    try:
        report = run_monte_carlo_simulation(
            team_a, team_b, neutral=neutral, n_sims=n_sims,
            missing_players_a=missing_players_a, missing_players_b=missing_players_b,
            knockout=knockout, city=city, country=country, referee=referee
        )
        return report
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=503, 
            detail="Models not trained. Please run the POST /api/v1/admin/train endpoint first, or execute the pipeline."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running simulation: {str(e)}")

# Admin / Maintenance Endpoints (protected by API Key in production, left simple for dev)
@app.post("/api/v1/admin/scrape", tags=["Admin"], dependencies=[Depends(verify_admin_key)])
def trigger_scraping(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Triggers the full scraping and ELO calculation pipeline in the background."""
    background_tasks.add_task(run_scraper_pipeline)
    return {"message": "Scraping pipeline started in background. This might take 1-2 minutes."}

@app.post("/api/v1/admin/train", tags=["Admin"], dependencies=[Depends(verify_admin_key)])
def trigger_training(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Triggers the ML model retraining pipeline in the background."""
    background_tasks.add_task(train_and_save_models)
    return {"message": "Model retraining started in background."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
