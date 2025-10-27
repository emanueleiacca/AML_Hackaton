/**
 * Daily Submission Tracker — Conditional Formatting & Setup
 * Usage:
 * 1) Import the CSV into a new Google Sheet.
 * 2) Extensions → Apps Script → paste this code → Run setupTracker().
 * 3) Approve permissions if prompted.
 */
function setupTracker() {
  const IMPROVE = 0.01;   // 🟢 threshold for meaningful improvement
  const NOISE   = 0.005;  // 🟡 leaderboard noise band (±NOISE)
  
  const ss = SpreadsheetApp.getActive();
  const sh = ss.getActiveSheet(); // assume CSV imported on first sheet
  
  // Freeze header row
  sh.setFrozenRows(1);
  
  // Nice number formats
  // val_MRR (I), LB_MRR (J), delta (K)
  sh.getRange("I:I").setNumberFormat("0.000");
  sh.getRange("J:J").setNumberFormat("0.000");
  sh.getRange("K:K").setNumberFormat("+0.000;-0.000;0.000");
  
  // Auto-resize columns
  sh.autoResizeColumns(1, sh.getLastColumn());
  
  // Conditional formatting for delta (column K)
  const rules = sh.getConditionalFormatRules();
  const lastRow = sh.getMaxRows();
  
  // Green: delta >= IMPROVE
  rules.push(SpreadsheetApp.newConditionalFormatRule()
    .whenFormulaSatisfied(`=$K2>=${IMPROVE}`)
    .setBackground("#C6EFCE")
    .setFontColor("#006100")
    .setRanges([sh.getRange(2, 11, lastRow-1, 1)]) // K2:K
    .build());
  
  // Yellow: ABS(delta) <= NOISE
  rules.push(SpreadsheetApp.newConditionalFormatRule()
    .whenFormulaSatisfied(`=ABS($K2)<=${NOISE}`)
    .setBackground("#FFEB9C")
    .setFontColor("#7F6000")
    .setRanges([sh.getRange(2, 11, lastRow-1, 1)])
    .build());
  
  // Red: delta <= -NOISE
  rules.push(SpreadsheetApp.newConditionalFormatRule()
    .whenFormulaSatisfied(`=$K2<=-${NOISE}`)
    .setBackground("#F8CBAD")
    .setFontColor("#9C0006")
    .setRanges([sh.getRange(2, 11, lastRow-1, 1)])
    .build());
  
  sh.setConditionalFormatRules(rules);
  
  // Optional: add a "Today" date to empty date cells in column A when editing
  addOnEditTrigger_();
}

/**
 * Adds an installable onEdit trigger to stamp today's date in column A
 * when a row is edited and date cell is empty.
 */
function addOnEditTrigger_() {
  const projectTriggers = ScriptApp.getProjectTriggers();
  const hasTrigger = projectTriggers.some(t => t.getHandlerFunction() === 'onEditStampDate');
  if (!hasTrigger) {
    ScriptApp.newTrigger('onEditStampDate').forSpreadsheet(SpreadsheetApp.getActive()).onEdit().create();
  }
}

function onEditStampDate(e) {
  try {
    const sh = e.range.getSheet();
    const row = e.range.getRow();
    if (row <= 1) return; // skip header
    const dateCell = sh.getRange(row, 1); // column A
    if (!dateCell.getValue()) {
      dateCell.setValue(new Date());
    }
  } catch (err) {
    // no-op
  }
}
