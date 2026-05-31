
const {ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings} = require('@langchain/google-genai');
const {StringOutputParser,CommaSeparatedListOutputParser,StructuredOutputParser} = require('@langchain/core/output_parsers');
const {ChatPromptTemplate} = require('@langchain/core/prompts');


const path = require('path');
const fs = require('fs')
// const {DocxLoader} = require('@langchain/community/')
const { DocxLoader } = require("@langchain/community/document_loaders/fs/docx");
const { createStuffDocumentsChain } = require("@langchain/classic/chains/combine_documents");

const {createRetrievalChain} = require('@langchain/classic/chains/retrieval')


const { PDFLoader } = require("@langchain/community/document_loaders/fs/pdf");
const {CheerioWebBaseLoader} = require('@langchain/community/document_loaders/web/cheerio');
const {RecursiveCharacterTextSplitter} = require('@langchain/textsplitters');
const {OpenAIEmbeddings,} = require('@langchain/openai');

const {HNSWLib} = require("@langchain/community/vectorstores/hnswlib")
const {MemoryVectorStore} = require('@langchain/classic/vectorstores/memory') // saved in  RAM for small docs



const {Document} = require('@langchain/core/documents')
const {z} = require('zod');
const { is } = require('zod/locales');

require('dotenv').config();



const llm = new ChatGoogleGenerativeAI({
    model: "gemini-2.5-flash",
    temperature:0.5,
    maxTokens:1000,
    verbose:false
})




// console.log(docs[0].metadata);    // Page number, source, etc.


const documentRag =async(docpath)=>{

    try {
        console.time('RAG TOOK')

        const prompt = ChatPromptTemplate.fromTemplate(`
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            
            Formatting Instructions : {format_instructions}
            context:{context}
            Question:{input}
        `)

        const outputParser = new StringOutputParser();
        // const outputParser = StructuredOutputParser.fromZodSchema(
        // z.object({
        //     concept: z.string(),
        //     advantages: z.array(z.string()),
        //     disadvantages: z.array(z.string())
        // })
        // );

        const chain = await createStuffDocumentsChain({llm,prompt});
        const isPdf =(path.extname(docpath) === '.pdf') // check doc extension
      
        console.log(isPdf,path.extname(docpath));
        
        let loader ;
        if(path.extname(docpath) === '.docx')loader =  new DocxLoader(docpath) //pdf DOCUMENTS
        if(path.extname(docpath) === '.pdf') loader = new PDFLoader(docpath, {splitPages: true, }) //pdf DOCUMENTS

  
        const docs = await loader.load();
        // console.log(docs[20]); // Text from page 1

        const splitter =  new RecursiveCharacterTextSplitter({
            chunkSize:1000,
            chunkOverlap:200,
        })

        const splitDocs = await splitter.splitDocuments(docs);

        const embeddings = new OpenAIEmbeddings();
        // const embeddings = new GoogleGenerativeAIEmbeddings({model: "text-embedding-004"})
        // const vectorstores = await MemoryVectorStore.fromDocuments(splitDocs,embeddings);
        //Chnaged from memory storage(RAM) to disk stotage
        var vectorStores ;
        
        let fileBasename =path.basename(docpath, path.extname(docpath));
        
        if (fs.existsSync(`./vector/vector-db/${fileBasename}`)) {
            vectorStores = await HNSWLib.load(`./vector/vector-db/${fileBasename}`,embeddings);
            console.log("FROM DISK SAVE")
           
        } else {
            vectorStores = await HNSWLib.fromDocuments( splitDocs,embeddings);

            // vectorStores = await HNSWLib.fromDocuments(splitDocs,embeddings);

            await vectorStores.save(`./vector/vector-db/${fileBasename}`);

             console.log(`NEW SAVE ${docpath.split('/')[1]}`)
        }
                
        
        
        
        
        
        
        const retriever = vectorStores.asRetriever({k:3});

        const retrievalChain =  await createRetrievalChain({
            combineDocsChain:chain,//takes the k(we used 5) chunks found and stuffs them into a prompt for the model/LLM
            retriever //search engine
        })
                          
        const resp = await retrievalChain.invoke({
            input:`Compare B-Trees and LSM Trees according to the book.`,
            format_instructions:outputParser
            // format_instructions:outputParser.getFormatInstructions()
            
            // context:docs
        })
        console.log(resp.answer)
        console.timeEnd('RAG TOOK')






    } catch (error) {
        console.error(error)
    }
}
// documentRag("files/Gresham_Norman_CV_Clean.pdf")
// documentRag("files/worddocument.docx")
documentRag('files/Designing_Data-Intensive_Applications_TH.pdf')
