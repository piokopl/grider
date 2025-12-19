"""
Compact Dashboard Template for Grid Bot
Clean, table-based layout with all essential information
"""

DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grid Bot Dashboard</title>
    <style>
        :root {
            --bg-dark: #0d1117;
            --bg-card: #161b22;
            --bg-hover: #21262d;
            --border: #30363d;
            --text: #c9d1d9;
            --text-dim: #8b949e;
            --green: #3fb950;
            --red: #f85149;
            --yellow: #d29922;
            --blue: #58a6ff;
            --purple: #a371f7;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
            background: var(--bg-dark);
            color: var(--text);
            font-size: 13px;
            line-height: 1.5;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 15px;
        }
        
        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 15px;
        }
        
        .header h1 {
            font-size: 18px;
            color: var(--green);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--green);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        
        .header-info {
            display: flex;
            gap: 20px;
            font-size: 12px;
            color: var(--text-dim);
        }
        
        .mode-badge {
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }
        
        .mode-live { background: rgba(248,81,73,0.2); color: var(--red); }
        .mode-paper { background: rgba(63,185,80,0.2); color: var(--green); }
        
        /* Summary Cards Row */
        .summary-row {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .summary-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
            text-align: center;
        }
        
        .summary-card .label {
            font-size: 10px;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .summary-card .value {
            font-size: 20px;
            font-weight: 600;
            margin-top: 4px;
        }
        
        .summary-card .sub {
            font-size: 11px;
            color: var(--text-dim);
            margin-top: 2px;
        }
        
        .positive { color: var(--green); }
        .negative { color: var(--red); }
        .neutral { color: var(--yellow); }
        
        /* Section */
        .section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            margin-bottom: 15px;
            overflow: hidden;
        }
        
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 15px;
            background: var(--bg-hover);
            border-bottom: 1px solid var(--border);
            cursor: pointer;
        }
        
        .section-header h2 {
            font-size: 13px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .section-toggle {
            color: var(--text-dim);
            font-size: 12px;
        }
        
        .section-content {
            padding: 0;
        }
        
        .section-content.collapsed {
            display: none;
        }
        
        /* Main Table */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        
        th {
            background: var(--bg-hover);
            padding: 8px 10px;
            text-align: left;
            font-weight: 600;
            color: var(--text-dim);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
        }
        
        td {
            padding: 8px 10px;
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
        }
        
        tr:hover {
            background: var(--bg-hover);
        }
        
        tr:last-child td {
            border-bottom: none;
        }
        
        .pair-cell {
            font-weight: 600;
            color: var(--blue);
        }
        
        .status-active {
            color: var(--green);
            font-size: 11px;
        }
        
        .status-stopped {
            color: var(--red);
            font-size: 11px;
        }
        
        .status-paused {
            color: var(--yellow);
            font-size: 11px;
        }
        
        .mini-bar {
            display: flex;
            gap: 1px;
            height: 16px;
            align-items: flex-end;
        }
        
        .mini-bar-segment {
            width: 4px;
            border-radius: 1px;
            min-height: 2px;
        }
        
        .mini-bar-buy { background: var(--green); opacity: 0.7; }
        .mini-bar-sell { background: var(--yellow); opacity: 0.7; }
        .mini-bar-current { background: white; }
        
        .action-btn {
            padding: 3px 8px;
            border-radius: 4px;
            border: 1px solid var(--border);
            background: transparent;
            color: var(--text);
            cursor: pointer;
            font-size: 10px;
            margin-right: 4px;
        }
        
        .action-btn:hover {
            background: var(--bg-hover);
        }
        
        .action-btn.danger:hover {
            background: rgba(248,81,73,0.2);
            border-color: var(--red);
            color: var(--red);
        }
        
        /* AI Badge */
        .ai-badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 9px;
            font-weight: 600;
        }
        
        .ai-bullish { background: rgba(63,185,80,0.2); color: var(--green); }
        .ai-bearish { background: rgba(248,81,73,0.2); color: var(--red); }
        .ai-neutral { background: rgba(210,153,34,0.2); color: var(--yellow); }
        
        /* Orders Table */
        .orders-mini {
            font-size: 11px;
        }
        
        .orders-mini td {
            padding: 5px 10px;
        }
        
        .order-buy { color: var(--green); }
        .order-sell { color: var(--red); }
        
        /* Grid Layout for Bottom Sections */
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        /* Exchange Compare */
        .compare-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
        }
        
        .compare-col {
            padding: 10px 15px;
        }
        
        .compare-col:first-child {
            border-right: 1px solid var(--border);
        }
        
        .compare-title {
            font-size: 11px;
            color: var(--text-dim);
            margin-bottom: 8px;
            text-transform: uppercase;
        }
        
        .compare-item {
            display: flex;
            justify-content: space-between;
            padding: 3px 0;
            font-size: 11px;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 15px;
            color: var(--text-dim);
            font-size: 11px;
        }
        
        /* Responsive */
        @media (max-width: 1200px) {
            .summary-row {
                grid-template-columns: repeat(3, 1fr);
            }
            .grid-2 {
                grid-template-columns: 1fr;
            }
        }
        
        @media (max-width: 768px) {
            .summary-row {
                grid-template-columns: repeat(2, 1fr);
            }
            .container {
                padding: 10px;
            }
            table {
                font-size: 11px;
            }
        }
        
        /* Scrollable table container */
        .table-container {
            overflow-x: auto;
            max-height: 400px;
            overflow-y: auto;
        }
        
        /* Profit highlight */
        .profit-cell {
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>
                <span class="status-dot"></span>
                Grid Bot
            </h1>
            <div class="header-info">
                <span id="mode-badge" class="mode-badge"></span>
                <span>Pairs: <strong id="pair-count">-</strong></span>
                <span>Updated: <span id="last-update">-</span></span>
            </div>
        </div>
        
        <!-- Summary Row -->
        <div class="summary-row">
            <div class="summary-card">
                <div class="label">Total P&L</div>
                <div class="value" id="total-pnl">-</div>
                <div class="sub" id="total-pnl-pct">-</div>
            </div>
            <div class="summary-card">
                <div class="label">Today P&L</div>
                <div class="value" id="today-pnl">-</div>
                <div class="sub" id="today-trades">- trades</div>
            </div>
            <div class="summary-card">
                <div class="label">Open Positions</div>
                <div class="value" id="open-positions">-</div>
                <div class="sub" id="open-value">$-</div>
            </div>
            <div class="summary-card">
                <div class="label">Pending Orders</div>
                <div class="value" id="pending-orders">-</div>
                <div class="sub" id="pending-split">- buy / - sell</div>
            </div>
            <div class="summary-card">
                <div class="label">Win Rate</div>
                <div class="value" id="win-rate">-</div>
                <div class="sub" id="win-loss">-W / -L</div>
            </div>
            <div class="summary-card">
                <div class="label">Balance</div>
                <div class="value" id="balance">-</div>
                <div class="sub" id="available">Available: $-</div>
            </div>
        </div>
        
        <!-- Account Info Row -->
        <div class="summary-row" style="margin-bottom: 15px;">
            <div class="summary-card">
                <div class="label">💰 Wallet Balance</div>
                <div class="value" id="wallet-balance">-</div>
                <div class="sub">USDT Futures</div>
            </div>
            <div class="summary-card">
                <div class="label">📊 Equity</div>
                <div class="value" id="equity">-</div>
                <div class="sub" id="equity-change">-</div>
            </div>
            <div class="summary-card">
                <div class="label">💵 Available</div>
                <div class="value" id="available-margin">-</div>
                <div class="sub">Free margin</div>
            </div>
            <div class="summary-card">
                <div class="label">🔒 Used Margin</div>
                <div class="value" id="used-margin">-</div>
                <div class="sub" id="margin-pct">-% of equity</div>
            </div>
            <div class="summary-card">
                <div class="label">📈 Unrealized PnL</div>
                <div class="value" id="unrealized-pnl">-</div>
                <div class="sub" id="roi-pct">ROI: -%</div>
            </div>
            <div class="summary-card">
                <div class="label">🎁 Bonus</div>
                <div class="value" id="bonus">-</div>
                <div class="sub">Trading bonus</div>
            </div>
        </div>
        
        <!-- Grids Table -->
        <div class="section">
            <div class="section-header" onclick="toggleSection('grids')">
                <h2>📊 Active Grids</h2>
                <span class="section-toggle" id="grids-toggle">▼</span>
            </div>
            <div class="section-content" id="grids-content">
                <div class="table-container">
                    <table id="grids-table">
                        <thead>
                            <tr>
                                <th>Pair</th>
                                <th>Status</th>
                                <th>Price</th>
                                <th>Range</th>
                                <th>Position</th>
                                <th>P&L</th>
                                <th>Trades</th>
                                <th>AI</th>
                                <th>Grid</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="grids-body">
                            <tr><td colspan="10" style="text-align:center;color:var(--text-dim)">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- Two Column Layout -->
        <div class="grid-2">
            <!-- Recent Orders -->
            <div class="section">
                <div class="section-header" onclick="toggleSection('orders')">
                    <h2>📝 Recent Fills</h2>
                    <span class="section-toggle" id="orders-toggle">▼</span>
                </div>
                <div class="section-content" id="orders-content">
                    <div class="table-container" style="max-height:250px">
                        <table class="orders-mini">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Pair</th>
                                    <th>Side</th>
                                    <th>Price</th>
                                    <th>Amount</th>
                                    <th>Profit</th>
                                </tr>
                            </thead>
                            <tbody id="orders-body">
                                <tr><td colspan="6" style="text-align:center;color:var(--text-dim)">Loading...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <!-- Exchange Sync -->
            <div class="section">
                <div class="section-header" onclick="toggleSection('exchange')">
                    <h2>🔄 Exchange Sync</h2>
                    <span class="section-toggle" id="exchange-toggle">▼</span>
                </div>
                <div class="section-content" id="exchange-content">
                    <div class="compare-grid">
                        <div class="compare-col">
                            <div class="compare-title">Bitget (Real)</div>
                            <div id="exchange-real">Loading...</div>
                        </div>
                        <div class="compare-col">
                            <div class="compare-title">Bot State (Local)</div>
                            <div id="exchange-local">Loading...</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- AI Analysis -->
        <div class="section">
            <div class="section-header" onclick="toggleSection('ai')">
                <h2>🤖 AI Analysis</h2>
                <span class="section-toggle" id="ai-toggle">▼</span>
            </div>
            <div class="section-content collapsed" id="ai-content">
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Pair</th>
                                <th>Sentiment</th>
                                <th>Score</th>
                                <th>Regime</th>
                                <th>Mode</th>
                                <th>Confidence</th>
                                <th>Summary</th>
                            </tr>
                        </thead>
                        <tbody id="ai-body">
                            <tr><td colspan="7" style="text-align:center;color:var(--text-dim)">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            Auto-refresh: 5s | Grid Bot v2.0
        </div>
    </div>

    <script>
        // Toggle sections
        function toggleSection(id) {
            const content = document.getElementById(id + '-content');
            const toggle = document.getElementById(id + '-toggle');
            content.classList.toggle('collapsed');
            toggle.textContent = content.classList.contains('collapsed') ? '▶' : '▼';
        }
        
        // Fetch helper
        async function fetchData(endpoint) {
            try {
                const response = await fetch(endpoint);
                return await response.json();
            } catch (e) {
                console.error('Fetch error:', e);
                return null;
            }
        }
        
        // Format helpers
        function fmt(n, d=2) {
            if (n === null || n === undefined || isNaN(n)) return '-';
            return Number(n).toFixed(d);
        }
        
        function fmtPnl(n) {
            if (n === null || n === undefined || isNaN(n)) return '-';
            const cls = n >= 0 ? 'positive' : 'negative';
            const sign = n >= 0 ? '+' : '';
            return `<span class="${cls}">${sign}$${fmt(n)}</span>`;
        }
        
        function fmtPct(n) {
            if (n === null || n === undefined || isNaN(n)) return '-';
            const cls = n >= 0 ? 'positive' : 'negative';
            const sign = n >= 0 ? '+' : '';
            return `<span class="${cls}">${sign}${fmt(n)}%</span>`;
        }
        
        function fmtTime(ts) {
            if (!ts) return '-';
            
            // If already in HH:MM format, return as-is
            if (typeof ts === 'string' && /^[0-9]{1,2}:[0-9]{2}$/.test(ts)) {
                return ts;
            }
            
            // Try to parse as date
            const d = new Date(ts);
            if (isNaN(d.getTime())) return ts || '-';
            
            return d.toLocaleTimeString('en-GB', {hour: '2-digit', minute: '2-digit'});
        }
        
        // Build mini grid visualization
        function buildMiniGrid(levels, currentPrice) {
            if (!levels || !levels.length) return '-';
            
            let html = '<div class="mini-bar">';
            const sortedLevels = [...levels].sort((a,b) => a.price - b.price);
            
            for (const level of sortedLevels.slice(0, 15)) {
                const height = 6 + Math.random() * 10;
                let cls = 'mini-bar-segment ';
                
                if (Math.abs(level.price - currentPrice) / currentPrice < 0.005) {
                    cls += 'mini-bar-current';
                } else if (level.price < currentPrice) {
                    cls += 'mini-bar-buy';
                } else {
                    cls += 'mini-bar-sell';
                }
                
                html += `<div class="${cls}" style="height:${height}px"></div>`;
            }
            
            html += '</div>';
            return html;
        }
        
        // Update dashboard
        async function updateDashboard() {
            // Fetch all data
            const [status, grids, stats, recentOrders] = await Promise.all([
                fetchData('/api/status'),
                fetchData('/api/grids'),
                fetchData('/api/stats'),
                fetchData('/api/recent_orders?limit=15')
            ]);
            
            // Update header
            if (status) {
                const modeBadge = document.getElementById('mode-badge');
                modeBadge.textContent = status.paper_trading ? 'PAPER' : 'LIVE';
                modeBadge.className = 'mode-badge ' + (status.paper_trading ? 'mode-paper' : 'mode-live');
                document.getElementById('pair-count').textContent = status.active_pairs || 0;
            }
            
            // Update summary cards
            if (stats) {
                document.getElementById('total-pnl').innerHTML = fmtPnl(stats.total_profit);
                document.getElementById('total-pnl-pct').innerHTML = fmtPct(stats.total_profit_pct);
                document.getElementById('today-pnl').innerHTML = fmtPnl(stats.today_profit);
                document.getElementById('today-trades').textContent = (stats.today_trades || 0) + ' trades';
                document.getElementById('open-positions').textContent = stats.open_positions || 0;
                document.getElementById('open-value').textContent = '$' + fmt(stats.open_value || 0, 0);
                document.getElementById('pending-orders').textContent = stats.pending_orders || 0;
                document.getElementById('pending-split').textContent = 
                    (stats.pending_buys || 0) + ' buy / ' + (stats.pending_sells || 0) + ' sell';
                
                const winRate = stats.total_trades > 0 
                    ? ((stats.winning_trades / stats.total_trades) * 100).toFixed(0) + '%'
                    : '-';
                document.getElementById('win-rate').textContent = winRate;
                document.getElementById('win-loss').textContent = 
                    (stats.winning_trades || 0) + 'W / ' + (stats.losing_trades || 0) + 'L';
            }
            
            // Update balance
            const balance = await fetchData('/api/exchange/balance');
            if (balance) {
                document.getElementById('balance').textContent = '$' + fmt(balance.total || 0, 0);
                document.getElementById('available').textContent = 'Available: $' + fmt(balance.available || 0, 0);
                
                // Account info row
                const walletBalance = balance.wallet_balance || (balance.total - (balance.unrealized_pnl || 0));
                document.getElementById('wallet-balance').textContent = '$' + fmt(walletBalance, 2);
                document.getElementById('equity').textContent = '$' + fmt(balance.total || 0, 2);
                document.getElementById('available-margin').textContent = '$' + fmt(balance.available || 0, 2);
                document.getElementById('used-margin').textContent = '$' + fmt(balance.used_margin || 0, 2);
                
                // Unrealized PnL with color
                const upnl = balance.unrealized_pnl || 0;
                const upnlEl = document.getElementById('unrealized-pnl');
                upnlEl.textContent = (upnl >= 0 ? '+$' : '-$') + fmt(Math.abs(upnl), 2);
                upnlEl.className = 'value ' + (upnl >= 0 ? 'positive' : 'negative');
                
                // ROI
                const roi = balance.roi_percent || 0;
                const roiEl = document.getElementById('roi-pct');
                roiEl.innerHTML = 'ROI: ' + (roi >= 0 ? '+' : '') + fmt(roi, 2) + '%';
                roiEl.style.color = roi >= 0 ? 'var(--green)' : 'var(--red)';
                
                // Margin percentage
                const marginPct = balance.total > 0 ? ((balance.used_margin || 0) / balance.total * 100) : 0;
                document.getElementById('margin-pct').textContent = fmt(marginPct, 1) + '% of equity';
                
                // Equity change (based on unrealized)
                const eqChange = document.getElementById('equity-change');
                if (upnl !== 0) {
                    eqChange.innerHTML = (upnl >= 0 ? '↑ +$' : '↓ -$') + fmt(Math.abs(upnl), 2);
                    eqChange.style.color = upnl >= 0 ? 'var(--green)' : 'var(--red)';
                }
                
                // Bonus
                document.getElementById('bonus').textContent = '$' + fmt(balance.bonus || 0, 2);
            }
            
            // Update grids table
            if (grids && grids.length > 0) {
                let html = '';
                for (const g of grids) {
                    const levels = await fetchData(`/api/levels/${g.pair}`);
                    const ai = await fetchData(`/api/ai/${g.pair}`);
                    
                    const statusCls = g.status === 'active' ? 'status-active' : 
                                     g.status === 'paused' ? 'status-paused' : 'status-stopped';
                    
                    const aiMode = ai?.final_recommendation?.grid_mode || ai?.recommendation?.grid_mode || 'neutral';
                    const aiBadgeCls = aiMode === 'long' ? 'ai-bullish' : 
                                       aiMode === 'short' ? 'ai-bearish' : 'ai-neutral';
                    
                    const rangeStr = `${fmt(g.range_low, g.current_price > 100 ? 0 : 4)} - ${fmt(g.range_high, g.current_price > 100 ? 0 : 4)}`;
                    
                    // Extract levels array from response
                    const levelsArray = levels?.levels || levels || [];
                    
                    html += `
                        <tr>
                            <td class="pair-cell">${g.pair.replace('USDT', '')}</td>
                            <td><span class="${statusCls}">● ${g.status}</span></td>
                            <td>$${fmt(g.current_price, g.current_price > 100 ? 2 : 4)}</td>
                            <td style="font-size:11px">${rangeStr}</td>
                            <td>${fmt(g.net_position, 4)}</td>
                            <td class="profit-cell">${fmtPnl(g.realized_profit)}</td>
                            <td>${g.total_trades || 0}</td>
                            <td><span class="ai-badge ${aiBadgeCls}">${aiMode.toUpperCase()}</span></td>
                            <td>${buildMiniGrid(levelsArray, g.current_price)}</td>
                            <td>
                                <button class="action-btn" onclick="rebuildGrid('${g.pair}')">↻</button>
                                <button class="action-btn danger" onclick="cancelOrders('${g.pair}')">✕</button>
                            </td>
                        </tr>
                    `;
                }
                document.getElementById('grids-body').innerHTML = html;
            }
            
            // Update orders table
            if (recentOrders && recentOrders.length > 0) {
                let html = '';
                for (const o of recentOrders) {
                    const sideCls = o.side === 'buy' ? 'order-buy' : 'order-sell';
                    const profit = o.profit || 0;
                    const profitCls = profit > 0 ? 'positive' : profit < 0 ? 'negative' : '';
                    const profitStr = profit !== 0 ? fmtPnl(profit) : '-';
                    
                    // Use display_time if available, otherwise format filled_at
                    const timeStr = o.display_time || fmtTime(o.filled_at);
                    
                    html += `
                        <tr>
                            <td>${timeStr}</td>
                            <td>${o.pair.replace('USDT', '')}</td>
                            <td class="${sideCls}">${o.side.toUpperCase()}</td>
                            <td>$${fmt(o.fill_price, o.fill_price > 100 ? 2 : 4)}</td>
                            <td>${fmt(o.fill_amount, 4)}</td>
                            <td class="${profitCls}">${profitStr}</td>
                        </tr>
                    `;
                }
                document.getElementById('orders-body').innerHTML = html;
            } else {
                document.getElementById('orders-body').innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-dim)">No recent fills</td></tr>';
            }
            
            // Update exchange sync
            const compare = await fetchData('/api/exchange/compare');
            
            if (compare && !compare.error) {
                let realHtml = '';
                let localHtml = '';
                
                // Real positions from Bitget
                const realPositions = compare.real?.position_details || [];
                
                for (const pos of realPositions) {
                    const sideCls = pos.side === 'long' ? 'positive' : 'negative';
                    const pnlCls = pos.pnl >= 0 ? 'positive' : 'negative';
                    realHtml += `<div class="compare-item">
                        <span>${pos.pair.replace('USDT','')}</span>
                        <span class="${sideCls}">${pos.side} ${fmt(pos.size, 4)}</span>
                        <span class="${pnlCls}">${pos.pnl >= 0 ? '+' : ''}$${fmt(pos.pnl, 2)}</span>
                    </div>`;
                }
                
                // Local positions from bot
                const localPositions = compare.local?.position_details || [];
                for (const pos of localPositions) {
                    const sideCls = pos.side === 'long' ? 'positive' : 'negative';
                    localHtml += `<div class="compare-item">
                        <span>${pos.pair.replace('USDT','')}</span>
                        <span class="${sideCls}">${pos.side} ${fmt(pos.size, 4)}</span>
                    </div>`;
                }
                
                // Summary
                const realCount = compare.real?.positions || 0;
                const localCount = compare.local?.positions || 0;
                const realOrders = compare.real?.open_orders || 0;
                
                if (realPositions.length === 0) {
                    realHtml = '<span style="color:var(--text-dim)">No positions</span>';
                }
                if (localPositions.length === 0) {
                    localHtml = '<span style="color:var(--text-dim)">No positions</span>';
                }
                
                // Add summary
                realHtml = `<div style="margin-bottom:8px;font-size:11px;color:var(--text-dim)">${realCount} positions, ${realOrders} orders</div>` + realHtml;
                localHtml = `<div style="margin-bottom:8px;font-size:11px;color:var(--text-dim)">${localCount} positions tracked</div>` + localHtml;
                
                document.getElementById('exchange-real').innerHTML = realHtml;
                document.getElementById('exchange-local').innerHTML = localHtml;
            } else if (compare && compare.error) {
                document.getElementById('exchange-real').innerHTML = `<span style="color:var(--red)">Error: ${compare.error}</span>`;
                document.getElementById('exchange-local').innerHTML = '';
            }
            
            // Update AI Analysis
            const aiData = await fetchData('/api/ai');
            if (aiData && aiData.ai_enabled && aiData.analyses) {
                let html = '';
                for (const pair of Object.keys(aiData.analyses).sort()) {
                    const ai = aiData.analyses[pair];
                    
                    const sentimentCls = ai.sentiment === 'bullish' || ai.sentiment === 'very_bullish' ? 'positive' :
                                        ai.sentiment === 'bearish' || ai.sentiment === 'very_bearish' ? 'negative' : '';
                    
                    const modeCls = ai.grid_mode === 'long' ? 'ai-bullish' :
                                   ai.grid_mode === 'short' ? 'ai-bearish' : 'ai-neutral';
                    
                    const score = ai.score || 0;
                    const scoreCls = score > 0 ? 'positive' : score < 0 ? 'negative' : '';
                    
                    html += `
                        <tr>
                            <td class="pair-cell">${pair.replace('USDT', '')}</td>
                            <td class="${sentimentCls}">${ai.sentiment || '-'}</td>
                            <td class="${scoreCls}">${score > 0 ? '+' : ''}${fmt(score, 2)}</td>
                            <td>${ai.regime || '-'}</td>
                            <td><span class="ai-badge ${modeCls}">${(ai.grid_mode || 'neutral').toUpperCase()}</span></td>
                            <td>${fmt((ai.confidence || 0) * 100, 0)}%</td>
                            <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${ai.summary || ''}">${ai.summary || '-'}</td>
                        </tr>
                    `;
                }
                document.getElementById('ai-body').innerHTML = html || '<tr><td colspan="7" style="text-align:center;color:var(--text-dim)">No AI data</td></tr>';
            } else {
                document.getElementById('ai-body').innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--text-dim)">AI not enabled</td></tr>';
            }
            
            // Update timestamp
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
        }
        
        // Actions
        async function rebuildGrid(pair) {
            if (confirm(`Rebuild grid for ${pair}?`)) {
                await fetch(`/api/grid/${pair}/rebuild`, {method: 'POST'});
                updateDashboard();
            }
        }
        
        async function cancelOrders(pair) {
            if (confirm(`Cancel all orders for ${pair}?`)) {
                await fetch(`/api/grid/${pair}/cancel_orders`, {method: 'POST'});
                updateDashboard();
            }
        }
        
        // Initial load and auto-refresh
        updateDashboard();
        setInterval(updateDashboard, 5000);
    </script>
</body>
</html>'''
