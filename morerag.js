const path = require('path');
const fs = require('fs')
let paths = 'files/how_to_create_pdf_Gresham_Norman_CV_Clean.docx'


console.log(path.extname(paths),path.extname(paths) === 'pdf')
console.log(fs.existsSync("./vector/vector-db"),"FILE EXISTS")


let cut = `files/Gresham_Norman_CV_Clean.pdf`

console.log(`${path.basename(cut, path.extname(cut))}`)
