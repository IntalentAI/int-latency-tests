package main

import (
	"fmt"
	"os"
	"strings"
	"time"

	"encoding/csv"
	"flag"
	"io/ioutil"
	"log"
	"math/rand"
	"path/filepath"

	"github.com/Microsoft/cognitive-services-speech-sdk-go/common"
	"github.com/Microsoft/cognitive-services-speech-sdk-go/speech"
	"github.com/joho/godotenv"
)

// Metrics represents performance measurements
type Metrics struct {
	Service    string  `json:"service" csv:"Service"`
	Region     string  `json:"region" csv:"Region"`
	InputText  string  `json:"input_text" csv:"Input Text"`
	TTFB       float64 `json:"ttfb_ms" csv:"TTFB (ms)"`
	E2ELatency float64 `json:"e2e_latency_ms" csv:"E2E Latency (ms)"`
	StartTime  float64 `json:"start_time" csv:"Start Time"`
	EndTime    float64 `json:"end_time" csv:"End Time"`
}

// SynthesisResult holds the audio data and metrics
type SynthesisResult struct {
	AudioData []byte
	Metrics   Metrics
}

// Create a struct to hold timing info
type synthesisTimings struct {
	startTime time.Time
	firstByte time.Time
	endTime   time.Time
}

func synthesizeStartedHandler(event speech.SpeechSynthesisEventArgs) {
	defer event.Close()
	fmt.Println("Synthesis started.")
}

func synthesizingHandler(event speech.SpeechSynthesisEventArgs) {
	defer event.Close()
	fmt.Printf("Synthesizing, audio chunk size %d.\n", len(event.Result.AudioData))
}

func synthesizedHandler(event speech.SpeechSynthesisEventArgs) {
	defer event.Close()
	fmt.Printf("Synthesized, audio length %d.\n", len(event.Result.AudioData))
}

func cancelledHandler(event speech.SpeechSynthesisEventArgs) {
	defer event.Close()
	fmt.Println("Received a cancellation.")
}

// synthesizeSpeech performs speech synthesis and returns audio data with metrics
func synthesizeSpeech(synthesizer *speech.SpeechSynthesizer, text string) (*SynthesisResult, error) {
	timings := &synthesisTimings{
		startTime: time.Now(),
	}

	// Create channels for synchronization
	firstByteChan := make(chan struct{})

	// Set up event handlers
	synthesizer.Synthesizing(func(evt speech.SpeechSynthesisEventArgs) {
		if timings.firstByte.IsZero() { // Only capture first byte once
			timings.firstByte = time.Now()
			close(firstByteChan)
		}
	})

	// Start synthesis
	task := synthesizer.SpeakTextAsync(text)

	// Wait for result or timeout
	var result speech.SpeechSynthesisOutcome
	select {
	case result = <-task:
		if result.Error != nil {
			return nil, fmt.Errorf("synthesis failed: %v", result.Error)
		}
	case <-time.After(30 * time.Second):
		return nil, fmt.Errorf("synthesis timed out")
	}
	defer result.Close()

	// Record end time
	timings.endTime = time.Now()

	// Wait for first byte or timeout
	select {
	case <-firstByteChan:
	case <-time.After(5 * time.Second):
		return nil, fmt.Errorf("no first byte received")
	}

	return &SynthesisResult{
		AudioData: result.Result.AudioData,
		Metrics: Metrics{
			Service:    "GoAzure",
			Region:     "westus",
			InputText:  text,
			TTFB:       float64(timings.firstByte.Sub(timings.startTime).Milliseconds()),
			E2ELatency: float64(timings.endTime.Sub(timings.startTime).Milliseconds()),
			StartTime:  float64(timings.startTime.Unix()),
			EndTime:    float64(timings.endTime.Unix()),
		},
	}, nil
}

// loadQuestions reads questions from file
func loadQuestions(filepath string) ([]string, error) {
	content, err := ioutil.ReadFile(filepath)
	if err != nil {
		return nil, err
	}
	questions := strings.Split(string(content), "\n")
	// Remove empty lines
	var filtered []string
	for _, q := range questions {
		if strings.TrimSpace(q) != "" {
			filtered = append(filtered, q)
		}
	}
	return filtered, nil
}

// Add function to write metrics to CSV
func writeMetricsToCSV(metrics []Metrics, outputDir string) error {
	file, err := os.Create(filepath.Join(outputDir, fmt.Sprintf("run_metrics_%s.csv",
		time.Now().Format("20060102_150405"))))
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	if err := writer.Write([]string{"Service", "Region", "Input Text", "TTFB (ms)",
		"E2E Latency (ms)", "Start Time", "End Time"}); err != nil {
		return err
	}

	// Write metrics
	for _, m := range metrics {
		if err := writer.Write([]string{
			m.Service,
			m.Region,
			m.InputText,
			fmt.Sprintf("%.2f", m.TTFB),
			fmt.Sprintf("%.2f", m.E2ELatency),
			fmt.Sprintf("%.6f", m.StartTime),
			fmt.Sprintf("%.6f", m.EndTime),
		}); err != nil {
			return err
		}
	}

	return nil
}

func main() {
	// Parse command line flags
	waitTime := flag.Int("wait", 0, "Wait time in seconds between iterations")
	iterations := flag.Int("iterations", 1, "Number of iterations to run")
	flag.Parse()

	// Load .env file
	if err := godotenv.Load(); err != nil {
		log.Fatal("Error loading .env file")
	}

	// Get credentials from .env file
	speechKey := os.Getenv("AZURE_SPEECH_API_KEY")
	speechRegion := os.Getenv("AZURE_SPEECH_REGION")

	if speechKey == "" || speechRegion == "" {
		log.Fatal("SPEECH_KEY and SPEECH_REGION must be set in .env file")
	}

	// Initialize speech configs
	speechConfig, err := speech.NewSpeechConfigFromSubscription(speechKey, speechRegion)
	if err != nil {
		log.Fatalf("Error creating speech config: %v", err)
	}
	defer speechConfig.Close()

	// Set output format for WAV
	speechConfig.SetProperty(common.SpeechServiceConnectionSynthOutputFormat, "riff-16khz-16bit-mono-pcm")
	speechConfig.SetSpeechSynthesisVoiceName("en-US-AvaMultilingualNeural")

	// Create synthesizer without audio output
	synthesizer, err := speech.NewSpeechSynthesizerFromConfig(speechConfig, nil)
	if err != nil {
		log.Fatalf("Error creating synthesizer: %v", err)
	}
	defer synthesizer.Close()

	// Load questions
	questions, err := loadQuestions("data/input/questions.txt")
	if err != nil {
		log.Fatalf("Error loading questions: %v", err)
	}

	// Create output directory with timestamp
	timestamp := time.Now().Format("20060102_150405")
	outputDir := filepath.Join("data", "output", timestamp)
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Fatalf("Error creating output directory: %v", err)
	}

	var allMetrics []Metrics

	// Process iterations
	for i := 0; i < *iterations; i++ {
		// Select random question
		questionIdx := rand.Intn(len(questions))
		question := questions[questionIdx]

		// Synthesize speech
		result, err := synthesizeSpeech(synthesizer, question)
		if err != nil {
			log.Printf("Error synthesizing speech: %v", err)
			continue
		}

		// Save audio file
		audioFilename := filepath.Join(outputDir, fmt.Sprintf("output_%d.wav", i))
		if err := ioutil.WriteFile(audioFilename, result.AudioData, 0644); err != nil {
			log.Printf("Error saving audio file: %v", err)
			continue
		}

		// Collect metrics in memory
		allMetrics = append(allMetrics, result.Metrics)

		// Wait if specified
		if *waitTime > 0 && i < *iterations-1 {
			time.Sleep(time.Duration(*waitTime) * time.Second)
		}
	}

	// Write all metrics to CSV at the end
	if err := writeMetricsToCSV(allMetrics, outputDir); err != nil {
		log.Printf("Error writing metrics to CSV: %v", err)
	}
}
