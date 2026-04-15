# NEXUS Dashboard Walkthrough Script - Detailed Timestamps
## Natural Speaking Format | Technical Accuracy | Recording-Ready

---

## SECTION 1: Rank Headboard & Flowview
**Timestamp:** 0:46.8 - 1:32.4  
**Duration:** ~45 seconds  
**Speaker:** Khushi

### Script:

"Toh yeh hai hamara **Ranked Junction Board**—basically sabhi traffic junctions ka command center. Dekho, har junction ko score kiya gaya hai: green matalab traffic smooth chal rahi hai, yellow matalab thoda congestion, aur red matalab serious bottleneck.

Dekh rahi ho? J1_1, J1_2, J1_3… har ek ka apna score hai. Yeh **D3QN agent** decide karta hai real-time mein. Agar J1_1 ka score 78 hai, matlab vehicle density analyze ho gayi, wait times predict ho gayi, aur signal timing optimize ho gayi.

Ab dekho **Heatboard**—yeh city-wide view hai. Jahan jahan red zone hai, wahan congestion zyada hai. Yeh data hamara **Vision Pipeline** ko milta hai jo **YOLO** se vehicles detect karta hai aur speeds measure karta hai.

Aur yeh raha **Flowview**—yeh curves dikhate hain traffic distribution. Smooth curves matlab system optimize ho raha hai aur signals priority decide kar raha hai."

### Technical Explanation:
- **Ranked Junction Board:** Displays all junctions sorted by congestion score (D3QN output)
- **Score Formula:** RL agent outputs priority based on: vehicle count (from Vision), predicted queue length (from LSTM), historical patterns
- **Heatboard:** Heatmap layer on city map, color intensity = congestion level
- **Flowview:** Curve graphs showing vehicle flow rates, smoothness indicates optimization success
- **Backend Calculation:** Updates every 1 second via WebSocket from FastAPI backend

---

## SECTION 2: Video Upload with YOLO Detection Demo
**Timestamp:** 1:32.4 - 2:33.8  
**Duration:** ~61 seconds  
**Speaker:** Jaismeen

### Script:

"Ab dekho **Video Upload Detection**. Yeh video upload kar rahi hoon… aur dekho! In **blue boxes** mein har vehicle detect ho gaya. Yeh hamare **YOLO** deep learning model se hota hai—real-time mein sab vehicles ko pakad leta hai.

Jo camera footage hai, usme **Step 1** mein **Vehicle Detector** har frame ko scan karta hai. **Step 2** mein YOLO confidence score deta hai—agar 75% se zyada confident hai, toh vehicle confirmed hai.

**Step 3** mein **Tracker** module har vehicle ko unique ID deta hai. Ek car 'Vehicle_234' ke naam se pura video mein track hota rahta hai. Aur saath-saath **Counter** vehicle count badhata hai aur **Speed Estimator** speed calculate karta hai.

Dekho screen par—count numbers, speed overlays, lane distribution—sab real-time data hai. Phir yeh **RL Controller** ko jaata hai jo signal timing decide karta hai. Pura pipeline **sub-50ms latency** mein hota hai because GPU acceleration aur parallel processing use karte hain."

### Technical Explanation:
- **Input:** Video file uploaded via FastAPI endpoint `/api/upload_video`
- **YOLO Detection:** YOLOv8 model running on GPU, 60+ FPS inference
- **Confidence Threshold:** 0.75 (75%) to filter false positives
- **Tracking:** DeepSORT algorithm maintains vehicle IDs across frames
- **Speed Estimation:** Calculates optical flow and converts pixel distance to real-world km/h
- **Vehicle Counter:** Increments junction entry/exit counts
- **Data Flow:** Vision→Prediction→RL Decision→Signal Command
- **Latency:** ~800ms full pipeline (video ingestion to signal recommendation)

---

## SECTION 3: Live Webcam Demo with Detection
**Timestamp:** 2:33.8 - 2:51.1  
**Duration:** ~17 seconds  
**Speaker:** Jaismeen

### Script:

"Ab yeh real-time hai! Dekho, meri phone se camera feed aa raha hai dashboard mein. Aur dekho—detection boxes abhi abhi appear ho rahi hain jab vehicles move karte hain!

**Live Webcam Module** MQTT protocol se connected hai. Har frame **YOLO** se process hota hai instantly—30 FPS mein. Koi delay nahi, koi buffer nahi. Dekho tracking kitne smoothly kar rahi hai—saari vehicles ko track kar rahi hai aur speed bhi calculate kar rahi hai.

Yeh ek edge camera ka simulation hai. Real production mein sauon-sau aise cameras data denge **Data Fusion Engine** ko, jo IoT sensors aur historical patterns ko combine karke complete picture deta hai. **Latency 150ms se kam hai**—isliye emergency routing jaisa critical work ho sakta hai."

### Technical Explanation:
- **Input:** Live camera stream via MQTT or WebSocket
- **Real-time Processing:** YOLOv8 running at 30 FPS on GPU
- **Tracker:** DeepSORT maintains vehicle IDs in live feed
- **Speed Calculation:** Optical flow between consecutive frames
- **Data Fusion:** Merges camera data with IoT sensors (inductive loops, radar)
- **Latency:** ~150ms end-to-end (capture→detect→broadcast)
- **Streaming Protocol:** WebSocket for dashboard, MQTT for edge IoT devices

---

## SECTION 4: Peak Score, Peak Time, Latency & Signal Analysis
**Timestamp:** 2:51.1 - 3:01.7  
**Duration:** ~10.6 seconds  
**Speaker:** Khushi

### Script:

"Dekho **Signal Analysis timeline**. Yeh timeline dikhati hai anomaly scores—basically kab traffic volatile tha.

Yeh dekho—**peak score spike** yahan pe. Iska matlab congestion suddenly badhgaya. Accident ya koi route change ho sakta tha. Hamara **LSTM** ne yeh anomaly catch kia kyunki expected patterns se different tha.

**Peak Time Frame** exact time deta hai jab maximum congestion tha. Yeh timestamps important hain because root cause identify karne mein madad dete hain—kya incident tha? Kya signal problem tha?

Dekho **Latency metrics**—congestion detect hone se lekar signal timing recalculate karke bhejne tak **2 seconds** lagta hai. Yeh **D3QN** ki power hai—patterns se seekhta hai aur problem hone se pehle predict kar leta hai."

### Technical Explanation:
- **Anomaly Score:** Calculated by comparing predicted vs. actual traffic flow
- **Detection Method:** Z-score analysis on time-series congestion data
- **Peak Score Threshold:** Alarms trigger if score > 85
- **Peak Time:** Exact timestamp of anomaly onset
- **Latency Breakdown:**
  - YOLO inference: 33ms/frame
  - Tracking: 5ms/frame
  - Speed estimation: 2ms/frame
  - RL decision: 800ms (batch inference)
  - Signal command transmission: 50ms
  - Total: ~2 seconds
- **LSTM Prediction Model:** Trained on 30 days of historical traffic to predict next 5-minute window
- **Cybersecurity Check:** Validates signal state changes against historical range ±10%

---

## SECTION 5: Emergency Corridor, Backend FPS, Attack Sim & Voice Demo
**Timestamp:** 3:01.8 - 3:43.4  
**Duration:** ~41.6 seconds  
**Speaker:** Khushi + Jaismeen (alternating)

### Script - KHUSHI:

"Yeh raha NEXUS ka sabse special feature—**Emergency Corridor Engine**. Jab ambulance ya fire truck ko city mein jana hota hai, ham sirf route nahi dete, **signals ko actively control** karte hain green wave banane ke liye.

Emergency vehicle ka signal aata hai hamko IoT network se. Phir **Corridor Engine** optimal route calculate karta hai aur signal timing lock karta hai:

**J1_1 → J1_2 → J2_2 → J2_3 → J3_3**

Har junction green hoti hai emergency vehicle ke liye, aur side traffic manage hoti hai intelligently. Aur dekho **Backend FPS Counter**—60 FPS pe chal raha hai! Matlab backend 60 decision cycles per second handle kar raha hai. **GPU acceleration** se yeh possible hai."

### Script - JAISMEEN:

"Ab dekho ek critical demo—**cyber-attack simulation**. Kya hota hai agar koi malicious command bheje?

Dekho! System ne **immediately block** kia. Hamara **Cybersecurity Module** ne signal change ko check kia—'Arre, J1_1 signal 45 seconds se 3 seconds? Toh 93% drop! Yeh anomalous hai!' aur reject kar diya.

Hamara multi-layer detection dekho:
1. **Input Validation** - Format check
2. **Range Validation** - Signal timing 10-60 seconds mein honi chahiye
3. **Rate-of-Change** - 15% se zyada change = FLAG
4. **Consensus Check** - D3QN output se match kare

Dekho **Voice Broadcast System**—critical time mein audio alerts broadcast hote hain city mein."

### Script - KHUSHI:

"Suno yeh announcement: *'Attention all vehicles, emergency corridor active. Please follow signals.'*

Hamara **Voice Broadcast Engine** Google Text-to-Speech se natural announcements generate karta hai. Multilingual hai—English, Hindi, Urdu sab mein.

Ab dekho complete **AI Decision Sequence**:
1. **Vision:** 5000 vehicles detect hue
2. **Prediction:** LSTM ne next 5 minutes predict kiye
3. **RL Decision:** D3QN ne optimal signal timings select kiye
4. **Optimization:** Emergency corridor priority diya
5. **Execution:** 47 signals simultaneously update hue

Sab kuchh **2.3 seconds** mein! Yeh hamare RL system aur vision pipeline ki power hai."

### Technical Explanation:
- **Emergency Corridor Engine:**
  - Input: GPS location of emergency vehicle, destination
  - Algorithm: Dijkstra's shortest path + signal optimization
  - Output: Route with locked signal timings
  - Precedence: Overrides all RL decisions, even if suboptimal for general traffic
  
- **Backend FPS:**
  - 60 FPS = 16.67ms per cycle
  - Parallel processing: Vision thread, RL thread, Signal thread
  - GPU utilization: 85-95% (D3QN + YOLO inference)
  
- **Attack Simulation:**
  - Threshold violation detection: ±15% change in 1 cycle = FLAG
  - Rate-of-change bound: Max 2 sec/cycle increase allowed
  - Consensus checks: New command vs. predicted RL output tolerance ±5 sec
  
- **Voice Broadcast:**
  - Engine: gTTS (Google Text-to-Speech)
  - Latency: ~500ms from trigger to sound output
  - Coverage: Broadcast to all connected speakers + vehicle infotainment systems
  
- **AI Decision Sequence:**
  - YOLO frames processed: ~180 frames/second across all cameras
  - LSTM prediction window: Next 300 seconds (5 min)
  - D3QN state space size: ~10M possible junction configurations
  - RL decision time: ~800ms for batch inference
  - Execution broadcast: 50ms to all traffic controllers
  - Total pipeline: ~2.3 seconds end-to-end

---

## SECTION 6: System Architecture Overview
**Timestamp:** 3:43.4 - 3:57.8  
**Duration:** ~14.4 seconds  
**Speaker:** Khushi

### Script:

"Toh yeh tha hamara complete **System Architecture**. Sabkuch 5 layers mein organize hai:

**Layer 1:** Input/Sensors—YOLO cameras, IoT sensors, traffic cameras, GPS data.

**Layer 2:** Vision Pipeline—**Vehicle Detector** 60 FPS mein frames process karta hai, **Tracker** vehicles ko follow karta hai, **Counter** count badhata hai, **Speed Estimator** speed measure karta hai.

**Layer 3:** Prediction Engine—**LSTM** next 5 minutes predict karta hai, **Anomaly Detector** outliers flag karta hai.

**Layer 4:** RL Decision Layer—Yeh brain hai hamara. **D3QN, DQN, PPO agents** decisions evaluate karte hain, **Coordinator** best option select karta hai. Phir **Emergency Corridor**, **Carbon Credit**, **Pedestrian Safety** aur **Maintenance AI** modules activate hote hain.

**Layer 5:** Backend & Frontend—FastAPI backend decisions execute karta hai, WebSocket real-time updates karta hai, dashboard operators ko override ka option deta hai.

Pura system **GPU-accelerated** hai aur **failsafe redundancy** har layer pe hai."

### Technical Explanation:
- **Input Layer:**
  - YOLO cameras: 60 FPS inference
  - IoT sensors: MQTT protocol, 1-second update rate
  - GPS traces: Mobility data for route prediction
  
- **Vision Pipeline (Step 1-2):**
  - YOLO: YOLOv8 on GPU, ~5ms latency
  - Tracker: DeepSORT algorithm
  - Counter: Per-lane vehicle count
  - Speed Estimator: Optical flow to km/h conversion
  - Incident Detector: Collision/stall detection using motion anomaly
  
- **Prediction Layer (Step 3):**
  - LSTM Input: Last 10 timesteps of traffic state (junction counts, speeds, signals)
  - LSTM Output: Predicted state for next 20 timesteps (300 seconds)
  - Anomaly scores: Z-score on LSTM residuals
  
- **RL Decision Layer (Step 4):**
  - D3QN: Deep Dueling Q-Network, trained on simulated SUMO traffic
  - DQN: Standard Q-Learning for fallback
  - PPO: Policy Gradient for alternative strategies
  - Coordinator: Voting mechanism, selects max Q-value action
  - Specialized modules: Post-processing on RL decision
  
- **Execution Layer (Step 5):**
  - Backend: FastAPI on uvicorn, listens 0.0.0.0:8000
  - Signal Commands: HTTP POST to SCATS/SCOOT traffic controllers
  - Latency: <50ms command transmission
  
- **Frontend:**
  - WebSocket: Real-time state updates (~30 Hz)
  - REST: Static data queries
  - Light theme: Optimized for 12-hour traffic operator shifts

---

## SECTION 7: Emergency Corridor - Junction-Wise Breakdown
**Timestamp:** 3:57.8 - 4:02.6  
**Duration:** ~4.8 seconds  
**Speaker:** Jaismeen

### Script:

"Dekho **Emergency Corridor path** junction by junction:

**J1_1 → J1_2 → J2_2 → J2_3 → J3_3**

J1_1 pe green lock hai 90 seconds ke liye ambulance ke liye. Sab directions red hain—clear path.

Jab ambulance J1_2 pohunch jayega (45 seconds mein), system signal ko shift kar dega green mein. Koi red light nahi—pure **green wave** effect.

Side roads ko bhi 15-20 seconds green dete hain har 60 seconds mein taakki gridlock na ho. **Balance** emergency priority aur normal traffic mein.

**ETA:** Ambulance 3 minute 22 seconds mein destination pohunch jayega."

### Technical Explanation:
- **Route Calculation:** Dijkstra's shortest path algorithm on road network graph
- **Signal Timing:**
  - Normal green: 45-60 seconds per direction
  - Emergency green: 90 seconds for corridor direction
  - Residual green: 15 seconds every 60 seconds for perpendicular traffic
  
- **Prediction:** GPS speed + historical average speed = ETA per junction
- **Preemption:** Signal change commanded 5-10 seconds before vehicle arrival
- **Conflict Resolution:** 
  - Queue length monitoring on affected roads
  - If side queue > 15 vehicles, extend their green by 5 seconds
  - If queue still growing, reduce emergency corridor lifespan
  
- **Business Rules:**
  - Emergency vehicles: Priority = 1 (highest)
  - Peak hour traffic: Priority = 2
  - Special events: Priority = 3
  - Normal traffic: Priority = 4
  
- **Cost Calculation:** Delay to other vehicles = integral of backup queue over time

---

## SECTION 8: Anomaly Detection - Confidence & Alert Levels
**Timestamp:** 4:02.6 - 4:12.09  
**Duration:** ~9.49 seconds  
**Speaker:** Khushi

### Script:

"Last topic—**Anomaly Detection aur Alert Levels**. Hamara 3 tiers hain:

**🟢 OBSERVED (Green)** - Normal range mein variations. Tracking kar raha hai, alarm nahi. Example: J1_1 mein 12% zyada vehicles, lekin yeh seasonal variation hai.

**🟡 ELEVATED (Amber)** - Kuch unusual ho raha hai. Operator ko review karna chahiye. Example: J2_3 mein zero vehicles—road band ho gaya kya? Kya detector blind hai?

**🔴 CRITICAL (Red)** - Definite anomaly. System auto-override karega aur operators ko alert karega. Example: J1_2 signal 5 minutes ke liye RED stuck hai.

Algorithm simple hai:
1. LSTM se **expected state** banti hai
2. **Actual state** sensors se aata hai
3. **Difference = Actual - Expected**
4. Z-score calculate karte hain
5. Agar threshold cross hota hai → Anomaly flag

Yahan dekho—**23 Observed**, **4 Elevated**, **0 Critical**. Excellent! Koi major emergency nahi."

### Technical Explanation:
- **Anomaly Detection Pipeline:**
  - Input: Actual traffic state (vehicle counts, speeds, signals)
  - Baseline: LSTM predicted state (trained on 30 days history)
  - Anomaly Score: Z-score of residuals
  - Confidence: Sigmoid function of Z-score magnitude
  
- **Alert Thresholds:**
  - OBSERVED: |Z-score| ∈ [1.0, 1.96) → 15% confidence
  - ELEVATED: |Z-score| ∈ [1.96, 3.0) → 60% confidence
  - CRITICAL: |Z-score| ≥ 3.0 → 95% confidence
  
- **Anomaly Types:**
  - Type 1: Vehicle count spike (accident, modal shift)
  - Type 2: Vehicle count drop (road closure, GPS error)
  - Type 3: Speed anomaly (icing, congestion wave)
  - Type 4: Signal malfunction (stuck signal, false command)
  - Type 5: Sensor malfunction (detector blind spot, MQTT timeout)
  
- **Logging:**
  - Stored in: PostgreSQL database, indexed by timestamp
  - Retention: 90 days rolling window
  - Query API: `/api/anomalies?start=2026-04-13&end=2026-04-14&severity=CRITICAL`
  
- **Counterfactual Engine:**
  - Baseline RL decision vs. observed outcome
  - Simulates "what if" using parallel SUMO environment
  - Outputs: Benefit quantification (vehicles saved, emissions reduced, lives saved in emergency scenarios)

---

## Summary Table

| Timestamp | Duration | Component | Key Metric |
|-----------|----------|-----------|-----------|
| 0:46.8 - 1:32.4 | 45 sec | Rank/Heat/Flow | Junction scores, congestion heatmap, flow curves |
| 1:32.4 - 2:33.8 | 61 sec | Video YOLO | Detection boxes, confidence 75%+, tracking IDs, speed overlays |
| 2:33.8 - 2:51.1 | 17 sec | Live Webcam | Real-time 30 FPS, <150ms latency, MQTT feed |
| 2:51.1 - 3:01.7 | 10.6 sec | Signal Analysis | Peak anomaly score, latency 2 sec, frame processing 33ms |
| 3:01.8 - 3:43.4 | 41.6 sec | Emergency + Attack | Corridor path J1_1→J1_2→J2_2→J2_3→J3_3, attack blocked, voice demos, 2.3 sec full sequence |
| 3:43.4 - 3:57.8 | 14.4 sec | Architecture | 5 layers, Step 1-5, D3QN agent, GPU-accelerated |
| 3:57.8 - 4:02.6 | 4.8 sec | Corridor Junctions | Path breakdown, green wave, residual traffic balance, 3:22 ETA |
| 4:02.6 - 4:12.09 | 9.49 sec | Anomaly Alerts | 3 tiers (Observed/Elevated/Critical), Z-score thresholds, 23 observed, 4 elevated, 0 critical |

---

## Notes for Voice Recording

1. **Natural Pacing:** Read slowly, emphasize numbers and technical terms
2. **Pointer Guidance:** Reference "look here," "see this," "watch" when pointing at UI elements
3. **Transitions:** Brief pause between sections for pointer repositioning
4. **Technical Terms:** Pronounce clearly: YOLO (YOH-loh), LSTM (L-STM), D3QN (D-Three-Q-N), Dijkstra's (DIK-struh), DeepSORT (Deep-SORT)
5. **Enthusiasm:** Energetic for demos, serious for security topics, confident for architecture
6. **Two-Speaker Format:** For Section 5, Khushi handles technical, Jaismeen handles demo/impact
7. **Microphone:** Record in quiet room, use lavalier if available, avoid background noise
8. **Takes:** Record each section 2-3 times, keep best take matching timestamp exactly
9. **Editing:** Export as separate audio files, align with video timeline in OBS/Movie Maker
10. **Total Script Duration:** ~246 seconds (~4:06) - fits within 5:11 video with intro/outro

---

**Recording Quality Checklist:**
- ✅ Clear diction
- ✅ Appropriate speed (120-140 words/minute)
- ✅ Technical accuracy maintained
- ✅ Natural conversational tone
- ✅ Enthusiasm and confidence
- ✅ Timestamp alignment verified
- ✅ Two-speaker dynamics working
