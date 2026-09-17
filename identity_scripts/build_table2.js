const {
  Document, Packer, Paragraph, TextRun, HeadingLevel,
  InsertedTextRun, DeletedTextRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, ShadingType, BorderStyle
} = require("docx");
const fs = require("fs");

const AUTHOR = "Claude (R6-excluded re-analysis)";
const DATE = "2026-07-27T00:00:00Z";
let idc = 1;
const T = (t, o = {}) => new TextRun({ text: String(t), ...o });
const DEL = (t) => new DeletedTextRun({ text: String(t), id: idc++, author: AUTHOR, date: DATE });
const INS = (t) => new InsertedTextRun({ text: String(t), id: idc++, author: AUTHOR, date: DATE });

const W = [2200, 1550, 1550, 1550, 1600];
const TOTAL = W.reduce((a, b) => a + b, 0);

// cell: if old===new -> plain; else deletion(old)+insertion(new)
function cell(oldV, newV, i, opts = {}) {
  let runs;
  if (oldV === newV || newV === undefined) runs = [T(oldV, opts.run)];
  else runs = [DEL(oldV), INS(newV)];
  return new TableCell({
    width: { size: W[i], type: WidthType.DXA },
    shading: opts.shading ? { type: ShadingType.CLEAR, fill: opts.shading } : undefined,
    children: [new Paragraph({ alignment: opts.align || AlignmentType.CENTER, children: runs })],
  });
}
const hdr = (t, i) => cell(t, t, i, { shading: "D9E2F3", run: { bold: true } });

// rows: [class, [oldP,newP],[oldR,newR],[oldF1,newF1],[oldSup,newSup]]
const rows = [
  ["Round 1", ["0.84","0.80"], ["0.83","0.86"], ["0.83","0.83"], ["2,073","788"]],
  ["Round 2", ["0.86","0.88"], ["0.83","0.81"], ["0.84","0.84"], ["2,331","2,331"]],
  ["Round 3", ["0.88","0.89"], ["0.92","0.93"], ["0.90","0.91"], ["1,746","1,746"]],
  ["Round 4", ["0.94","0.94"], ["0.95","0.95"], ["0.94","0.95"], ["6,918","6,918"]],
  ["Round 5", ["0.84","0.86"], ["0.81","0.84"], ["0.83","0.85"], ["922","922"]],
];

const tableRows = [];
tableRows.push(new TableRow({ tableHeader: true, children:
  ["Class","Precision","Recall","F1-score","Support"].map((h,i)=>hdr(h,i)) }));
for (const r of rows) {
  tableRows.push(new TableRow({ children: [
    cell(r[0], r[0], 0, { align: AlignmentType.LEFT, run:{bold:true} }),
    cell(r[1][0], r[1][1], 1),
    cell(r[2][0], r[2][1], 2),
    cell(r[3][0], r[3][1], 3),
    cell(r[4][0], r[4][1], 4),
  ]}));
}
// Balanced accuracy row (precision/recall/f1 blank, value under F1 as in original)
tableRows.push(new TableRow({ children: [
  cell("Balanced accuracy","Balanced accuracy",0,{align:AlignmentType.LEFT,run:{bold:true}}),
  cell("","",1), cell("","",2),
  cell("0.90","0.88",3,{}),
  cell("13,990","12,705",4),
]}));
tableRows.push(new TableRow({ children: [
  cell("Macro avg","Macro avg",0,{align:AlignmentType.LEFT,run:{bold:true}}),
  cell("0.87","0.87",1),
  cell("0.86","0.88",2),
  cell("0.86","0.88",3),
  cell("13,990","12,705",4),
]}));

const table = new Table({
  columnWidths: W,
  width: { size: TOTAL, type: WidthType.DXA },
  rows: tableRows,
});

const doc = new Document({
  creator: AUTHOR,
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 } } },
    children: [
      new Paragraph({ spacing:{after:120}, children:[
        T("Table 2. ", {bold:true}),
        T("classification performance of random forest model (waveform 8–30)", {bold:true}),
      ]}),
      new Paragraph({ spacing:{after:180}, children:[
        INS("[Updated: Round 6 excluded — model now trained and tested on five individuals, R1–R5.]"),
      ]}),
      table,
      new Paragraph({ spacing:{before:180}, children:[
        T("Class-specific performance metrics for five individuals (rounds 1 - 5) showing precision, recall, F1-score, and support (number of test samples).", {italics:true}),
      ]}),
      new Paragraph({ children:[
        DEL("** Round 1 included a single repetition (i.e., three sessions: session 1, session 2, and session 3)."),
        INS("** Round 1 is represented by a single recording session in the analyzed dataset (R1S1), which reduces its test support relative to the earlier three-session version."),
      ]}),
    ],
  }],
});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync("Table2_classification_report_R6excluded_tracked.docx", b);
  console.log("wrote Table2 docx", b.length, "bytes");
});
