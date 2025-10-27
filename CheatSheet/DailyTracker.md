Daily Submission Tracker — Quick Start

FILES:
1) daily_submission_tracker.csv
2) sheets_conditional_formatting.gs  (Google Apps Script to color the Δ column)

HOW TO USE (Google Sheets):
1. Open Google Drive → New → Google Sheets.
2. File → Import → Upload → select daily_submission_tracker.csv →
   - Import location: "Replace spreadsheet" (or "Insert new sheet" if you already have a sheet)
   - Separator: Comma
   - Detect automatically
3. Go to Extensions → Apps Script → paste the contents of sheets_conditional_formatting.gs and Save.
4. Click Run ▶ setupTracker() → authorize if prompted.
   - This will: freeze header, format MRR/Δ columns, and add green/yellow/red conditional formatting on Δ.
5. Fill in val_MRR and LB_MRR after each submission; Δ computes automatically.

COLUMNS (A→N):
A date | B model_id | C pooling | D alpha_cos | E beta_mse | F gamma_infonce | G align_loss | H arch | I val_MRR | J LB_MRR | K delta (=J-I) | L params_M | M latency_ms | N notes

COLOR RULES:
- Green (🟢): Δ ≥ +0.01 → meaningful improvement
- Yellow (🟡): |Δ| ≤ 0.005 → within leaderboard noise
- Red (🔴): Δ ≤ −0.005 → regression
