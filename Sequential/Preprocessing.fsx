#r @".\packages\CsvHelper\lib\net45\CsvHelper.dll"
#r @".\packages\morelinq\lib\net40\MoreLinq.dll"
#r @".\packages\HtmlAgilityPack\lib\Net45\HtmlAgilityPack.dll"

open System
open System.IO
open System.Collections.Generic
open CsvHelper
open HtmlAgilityPack

let sourcePath = __SOURCE_DIRECTORY__

let saveIndexToTopicMappings path filename =
    use csvFile = new CsvWriter(new StreamWriter(filename: string))
    csvFile.Configuration.HasHeaderRecord <- false
    let topics = Directory.EnumerateDirectories(path, "*", SearchOption.TopDirectoryOnly)
    topics
    |> Seq.map (DirectoryInfo >> (fun dirInfo -> dirInfo.Name))
    |> Seq.indexed
    |> csvFile.WriteRecords

let loadIndexToTopicMappings filename =
    use csvFile = new CsvReader(new StreamReader(filename: string))
    csvFile.Configuration.HasHeaderRecord <- false
    let records = csvFile.GetRecords<int * string>() |> Array.ofSeq
    let straight = records |> dict
    let reversed = records |> Array.map (fun (index, topic) -> topic, index) |> dict
    straight, reversed

let docsReader contentLengthThreshold path =
    let directories = Directory.EnumerateDirectories(path, "*", SearchOption.TopDirectoryOnly)
    let files =
        directories
        |> Seq.collect (fun topicDir -> Directory.EnumerateFiles(topicDir, "*.txt", SearchOption.TopDirectoryOnly))
    seq {
        for file in files do
            let lines = File.ReadAllLines(file)
            let contentLength = lines |> Array.sumBy (fun line -> line.Length)
            if contentLength >= contentLengthThreshold then
                yield! lines
    }

let htmlSanitizer htmlString =
    let doc = new HtmlDocument()
    doc.LoadHtml(htmlString)
    let documenttNode = doc.DocumentNode
    let textNodes = documenttNode.SelectNodes("//text()[normalize-space()]")
    if isNull textNodes then
        String.Empty
    else
        textNodes
        |> Seq.map (fun node -> node.InnerText)
        |> Seq.filter (fun text -> (String.IsNullOrEmpty(text) |> not) && (String.IsNullOrWhiteSpace(text) |> not))
        |> String.concat " "

let simpleTokenizer (sanitizer: string -> string) (text: string) =
    // Refer to ASCII table
    let punctuations = 
        [| 
            yield! [| '\x00' .. '\x1F' |]
            yield! [| '\x21' .. '\x40' |]
            yield! [| '\x5B' .. '\x60' |]
            yield! [| '\x7B' .. '\x7F' |]
        |]
    let rawText = sanitizer text
    let toSplit =
        punctuations
        |> Array.fold (fun (prevState: string) punctuation -> prevState.Replace(punctuation, ' ')) rawText
    toSplit.Split([| ' ' |], StringSplitOptions.RemoveEmptyEntries)
    |> Array.map (fun token -> token.ToLower())

let loadStopWords filename =
    File.ReadAllLines(filename)
    |> Array.map (fun line -> line.Trim())
    |> HashSet

let addTokenToCounter (counter: Dictionary<string, int>) token =
        if counter.ContainsKey(token) then
            counter.[token] <- counter.[token] + 1
        else
            counter.Add(token, 0)

let saveVocabulary (texts: string seq) (tokenizer: string -> string array) (stopWordsSet: HashSet<string>) vocabularySize filename =
    let counter = new Dictionary<string, int>()
    texts
    |> Seq.iter (fun text ->
        text
        |> tokenizer
        |> Array.iter (addTokenToCounter counter))
    let validTokens = 
        counter
        |> Seq.filter (fun kv -> stopWordsSet.Contains(kv.Key) |> not)
        |> Seq.sortByDescending (fun kv -> kv.Value)
        |> Array.ofSeq
    let highFrequencyTokens =
        validTokens
        |> Array.take vocabularySize
        |> Array.map (fun kv -> kv.Key)
        |> Array.append [| "__UNK__" |]
        |> Array.ofSeq
    File.WriteAllLines(filename, highFrequencyTokens)

let fileFeeder shuffle topicsFile path =
    let _, reversed = loadIndexToTopicMappings topicsFile
    let prev =
        reversed
        |> Seq.collect (fun kv -> 
            let topicPath = Path.Combine(path, kv.Key)
            let filenames = Directory.EnumerateFiles(topicPath, "*.txt", SearchOption.TopDirectoryOnly)
            seq {
                for filename in filenames do
                    yield (filename, kv.Value)
            })
        |> Array.ofSeq
    if shuffle then
        MoreLinq.MoreEnumerable.RandomSubset(prev, prev.Length)
        |> Array.ofSeq
    else
        prev

let loadVocabulary vocabularyFile =
    File.ReadAllLines(vocabularyFile)
    |> Array.indexed
    |> Array.map (fun (index, token) -> token, index)
    |> dict

let mkSequence (tokenizer: string -> string array) (vocabularyDict: IDictionary<string, int>) (lines: string array) =
    let unkTokenIndex = vocabularyDict.["__UNK__"]
    let maxIndex = vocabularyDict.Count - 1
    [|
        for line in lines do
            for token in (tokenizer line) do
                match vocabularyDict.TryGetValue(token) with
                | true, tokenIndex -> yield single tokenIndex / single maxIndex
                | false, _ -> yield single unkTokenIndex / single maxIndex
    |]

let buildCrossValidationTrainingSet (fileFeed: (string * int) seq) contentLengthThreshold (mkSequence: string array -> 'Numeric array) testSetSize validationSetSize path =
    let skipSize = testSetSize + validationSetSize
    let transform (files: (string * int) seq) =
        files
        |> Seq.map (fun (filename, topicIndex) -> File.ReadAllLines(filename), topicIndex)
        |> Seq.filter (fun (lines, _) -> (lines |> Array.sumBy (fun line -> line.Length) >= contentLengthThreshold))
        |> Seq.map (fun (lines, topicIndex) ->
            lines
            |> mkSequence
            |> (fun sequence -> sequence, topicIndex))
    let writeSequencesToCsvFile (dataset: seq<'Numeric array * int>) (csvFilePath: string) =
        let xFilePath, yFilePath = csvFilePath + "_X.csv", csvFilePath + "_y.csv"
        let sequneces =
            dataset
            |> Seq.map (fun (sequnece, _) -> String.Join(",", sequnece))
        let labels =
            dataset
            |> Seq.map (fun (_, label) -> string label)
        File.WriteAllLines(xFilePath, sequneces)
        File.WriteAllLines(yFilePath, labels)
    let testSet =
        fileFeed
        |> Seq.take testSetSize
        |> transform
    let validationSet =
        fileFeed
        |> Seq.skip testSetSize
        |> Seq.take validationSetSize
        |> transform
    let trainSet =
        fileFeed
        |> Seq.skip skipSize
        |> transform
        |> Array.ofSeq
    writeSequencesToCsvFile testSet (Path.Combine(path, "test"))
    writeSequencesToCsvFile validationSet (Path.Combine(path, "validation"))
    writeSequencesToCsvFile trainSet (Path.Combine(path, "train"))

// Start processing
let docsPath = Path.Combine(sourcePath, "classifiedText")
let topicsFile = Path.Combine(sourcePath, "topics.csv")

let vocabularySize = 20000
let vocabularyFile = Path.Combine(sourcePath, "vocabulary.txt")
let stopWordsFile = Path.Combine(sourcePath, "stopWords.txt")

let contentLengthThreshold = 1000

let testSetSize = 3000
let validationSetSize = 1000

saveIndexToTopicMappings
    docsPath 
    topicsFile

saveVocabulary 
    (docsReader contentLengthThreshold docsPath) 
    (simpleTokenizer htmlSanitizer)
    (loadStopWords stopWordsFile)
    vocabularySize
    vocabularyFile

buildCrossValidationTrainingSet 
    (fileFeeder true topicsFile docsPath)
    contentLengthThreshold
    (mkSequence (simpleTokenizer htmlSanitizer) (loadVocabulary vocabularyFile))
    testSetSize
    validationSetSize
    sourcePath