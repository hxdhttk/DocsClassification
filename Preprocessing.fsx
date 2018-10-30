#r @".\packages\CsvHelper\lib\net45\CsvHelper.dll"
#r @".\packages\morelinq\lib\net40\MoreLinq.dll"

open System
open System.IO
open System.Collections.Generic
open CsvHelper

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

let simpleTokenizer (text: string) =
    // Refer to ASCII table
    let punctuations = 
        [| 
            yield! [| '\x00' .. '\x1F' |]
            yield! [| '\x21' .. '\x40' |]
            yield! [| '\x5B' .. '\x60' |]
            yield! [| '\x7B' .. '\x7F' |]
        |]
    let toSplit =
        punctuations
        |> Array.fold (fun (prevState: string) punctuation -> prevState.Replace(punctuation, ' ')) text
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

let saveVocabulary (texts: string seq) (tokenizer: string -> string array) (stopWordsSet: HashSet<string>) vocabularyFrequencyThreshold filename =
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
    let validTokensLength = validTokens.Length
    let vocabularySize = int ((single validTokensLength) * vocabularyFrequencyThreshold)
    let highFrequencyTokens =
        validTokens
        |> Array.take vocabularySize
        |> Array.map (fun kv -> kv.Key)
        |> Array.append [| "__UNK__" |]
        |> Array.ofSeq
    File.WriteAllLines(filename, highFrequencyTokens)
    highFrequencyTokens.Length

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

let inline simpleNormalizer vector =
    let sum = vector |> Array.sum
    vector |> Array.map (fun element -> element / sum)

let loadVocabulary vocabularyFile =
    File.ReadAllLines(vocabularyFile)
    |> Array.indexed
    |> Array.map (fun (index, token) -> token, index)
    |> dict

let vectorizer (tokenizer: string -> string array) (vocabulary: IDictionary<string, int>) normalizer (lines: string array) =
    let vectorDimension = vocabulary.Count
    let vector = Array.init vectorDimension (fun _ -> 0.0f)
    let counter = Dictionary<string, int>()
    lines
    |> Array.collect tokenizer
    |> Array.iter (addTokenToCounter counter)
    let unkIndex = vocabulary.["__UNK__"]
    for kv in counter do
        match vocabulary.TryGetValue(kv.Key) with
        | true, elementIndex ->
            vector.[elementIndex] <- single kv.Value
        | false, _ ->
            vector.[unkIndex] <- vector.[unkIndex] + single kv.Value
    vector |> normalizer

let buildCrossValidationTrainingSet (fileFeed: (string * int) seq) contentLengthThreshold (vectorizer: string array -> single array) testSetSize validationSetSize path =
    let skipSize = testSetSize + validationSetSize
    let transform (files: (string * int) seq) =
        files
        |> Seq.map (fun (filename, topicIndex) -> File.ReadAllLines(filename), topicIndex)
        |> Seq.filter (fun (lines, _) -> (lines |> Array.sumBy (fun line -> line.Length) >= contentLengthThreshold))
        |> Seq.map (fun (lines, topicIndex) ->
            lines
            |> vectorizer
            |> (fun vector -> Array.append vector [| single topicIndex |]))
    let writeVectorsToCsvFile (dataset: seq<single []>) (csvFilePath: string) =
        dataset
        |> Seq.map (fun vector -> String.Join(",", vector))
        |> (fun vectorStrings -> File.WriteAllLines(csvFilePath, vectorStrings))
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
    writeVectorsToCsvFile testSet (Path.Combine(path, "test.csv"))
    writeVectorsToCsvFile validationSet (Path.Combine(path, "validation.csv"))
    writeVectorsToCsvFile trainSet (Path.Combine(path, "train.csv"))
    trainSet.Length

// Start processing
let docsPath = Path.Combine(sourcePath, "classifiedText")
let topicsFile = Path.Combine(sourcePath, "topics.csv")

let vocabularyFrequencyThreshold = 0.025f
let vocabularyFile = Path.Combine(sourcePath, "vocabulary.txt")
let stopWordsFile = Path.Combine(sourcePath, "stopWords.txt")

let contentLengthThreshold = 1000

let testSetSize = 3000
let validationSetSize = 1000

saveIndexToTopicMappings
    docsPath 
    topicsFile

let vocabularySize =
    saveVocabulary 
        (docsReader contentLengthThreshold docsPath) 
        simpleTokenizer
        (loadStopWords stopWordsFile)
        vocabularyFrequencyThreshold
        vocabularyFile
printfn "Vocabulary Size: %d" vocabularySize

let trainSetLines =
    buildCrossValidationTrainingSet 
        (fileFeeder true topicsFile docsPath)
        contentLengthThreshold
        (vectorizer simpleTokenizer (loadVocabulary vocabularyFile) simpleNormalizer)
        testSetSize
        validationSetSize
        sourcePath
printfn "Train Set Lines: %d" trainSetLines