// src/pages/Yoga/Yoga.js
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';
import React, { useRef, useState, useEffect } from 'react';
import '@tensorflow/tfjs-backend-webgl';
import Webcam from 'react-webcam';
import { count } from '../../utils/music';

import Instructions from '../../components/Instrctions/Instructions';
import './Yoga.css';
import DropDown from '../../components/DropDown/DropDown';
import { poseImages } from '../../utils/pose_images';
import { POINTS, keypointConnections } from '../../utils/data';
import { drawPoint, drawSegment } from '../../utils/helper';

let skeletonColor = 'rgb(255,255,255)';
let poseList = [
  'Tree', 'Chair', 'Cobra', 'Warrior', 'Dog',
  'Shoulderstand', 'Traingle'
];

let interval;
let flag = false; // pose-detected flag

// ---------- Multilingual strings ----------
const STRINGS = {
  en: {
    correct: "Good — hold it.",
    raiseRight: "Raise your right hand up",
    raiseLeft: "Raise your left hand up",
    liftLegToThigh: "Lift your foot and place it on the opposite thigh",
    bendKneesMore: "Bend your knees more",
    straightenArm: "Straighten your arms",
    liftChest: "Lift your chest",
    bendFrontKnee: "Bend your front knee more",
    liftHips: "Lift your hips higher",
    keepLegsStraight: "Keep your legs straight",
    alignHipsShoulders: "Stack your shoulders over hips",
    reachTopArm: "Reach the top arm straight up",
  },
  ta: {
    correct: "நன்று — பிடித்து வையுங்கள்.",
    raiseRight: "உங்கள் வலது கையை மேலே தூக்கவும்",
    raiseLeft: "உங்கள் இடது கையை மேலே தூக்கவும்",
    liftLegToThigh: "கால்களை எடுத்து எதிரான இடுப்பு மீது வையுங்கள்",
    bendKneesMore: "முழங்கால்களை மேலும் வளைத்துக் கொள்ளுங்கள்",
    straightenArm: "கைகளை நேராக்கவும்",
    liftChest: "உங்கள் மார்பை உயர்த்துங்கள்",
    bendFrontKnee: "முன் முழங்காலை மேலும் வளைத்துக் கொள்ளுங்கள்",
    liftHips: "இடுப்பை மேலே தூக்குங்கள்",
    keepLegsStraight: "கால்களை நேராக வைத்துக் கொள்ளுங்கள்",
    alignHipsShoulders: "தோள்களை இடுப்பின் மேல் சென்னு ஒழுங்கு செய்யவும்",
    reachTopArm: "மேல்துறை கையை நேராக மேலே நீட்டிக்கவும்",
  }
};

// ---------- Text-to-Speech helpers ----------
let lastSpokenAt = 0;
let lastSpokenText = '';
function chooseVoice(lang) {
  const voices = window.speechSynthesis.getVoices() || [];
  if (!voices.length) return null;
  if (lang === 'ta') {
    const v = voices.find(v => v.lang.toLowerCase().startsWith('ta'));
    if (v) return v;
  }
  // prefer en
  return (voices.find(v => v.lang.toLowerCase().startsWith('en')) || voices[0]);
}
function speakText(text, lang = 'en') {
  if (!text) return;
  const now = Date.now();
  // throttle and de-duplicate
  if (text === lastSpokenText && (now - lastSpokenAt) < 2200) return;
  if ((now - lastSpokenAt) < 900) return; // small guard
  lastSpokenAt = now;
  lastSpokenText = text;

  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = (lang === 'ta') ? 'ta-IN' : 'en-US';
  const voice = chooseVoice(lang);
  if (voice) utter.voice = voice;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utter);
}

// ---------- Pose normalization & embedding (keeps your classifier compatibility) ----------
function get_center_point(landmarks, left_bodypart, right_bodypart) {
  let left = tf.gather(landmarks, left_bodypart, 1);
  let right = tf.gather(landmarks, right_bodypart, 1);
  const center = tf.add(tf.mul(left, 0.5), tf.mul(right, 0.5));
  return center;
}
function get_pose_size(landmarks, torso_size_multiplier = 2.5) {
  let hips_center = get_center_point(landmarks, POINTS.LEFT_HIP, POINTS.RIGHT_HIP);
  let shoulders_center = get_center_point(landmarks, POINTS.LEFT_SHOULDER, POINTS.RIGHT_SHOULDER);
  let torso_size = tf.norm(tf.sub(shoulders_center, hips_center));
  let pose_center_new = get_center_point(landmarks, POINTS.LEFT_HIP, POINTS.RIGHT_HIP);
  pose_center_new = tf.expandDims(pose_center_new, 1);
  pose_center_new = tf.broadcastTo(pose_center_new, [1, 17, 2]);
  let d = tf.gather(tf.sub(landmarks, pose_center_new), 0, 0);
  let max_dist = tf.max(tf.norm(d, 'euclidean', 0));
  let pose_size = tf.maximum(tf.mul(torso_size, torso_size_multiplier), max_dist);
  return pose_size;
}
function normalize_pose_landmarks(landmarks) {
  let pose_center = get_center_point(landmarks, POINTS.LEFT_HIP, POINTS.RIGHT_HIP);
  pose_center = tf.expandDims(pose_center, 1);
  pose_center = tf.broadcastTo(pose_center, [1, 17, 2]);
  landmarks = tf.sub(landmarks, pose_center);
  let pose_size = get_pose_size(landmarks);
  landmarks = tf.div(landmarks, pose_size);
  return landmarks;
}
function landmarks_to_embedding(landmarksArray) {
  // landmarksArray: [[x,y], ... 17 items]
  // convert -> tensor shape [1,17,2] then normalize and reshape to [1,34]
  try {
    const t = tf.tensor(landmarksArray).reshape([17, 2]); // [17,2]
    const expanded = tf.expandDims(t, 0); // [1,17,2]
    const normalized = normalize_pose_landmarks(expanded); // tf ops
    const embedding = tf.reshape(normalized, [1, 34]); // [1,34]
    return embedding;
  } catch (e) {
    // fallback: return zeros
    return tf.zeros([1, 34]);
  }
}

// ---------- Angle helpers (for rule checking) ----------
const toDeg = (r) => (r * 180) / Math.PI;
function angleBetween(a, b, c) {
  if (!a || !b || !c) return null;
  const abx = a.x - b.x, aby = a.y - b.y;
  const cbx = c.x - b.x, cby = c.y - b.y;
  const dot = abx * cbx + aby * cby;
  const magAB = Math.hypot(abx, aby);
  const magCB = Math.hypot(cbx, cby);
  if (magAB === 0 || magCB === 0) return null;
  let cos = dot / (magAB * magCB);
  cos = Math.min(1, Math.max(-1, cos));
  return toDeg(Math.acos(cos));
}
const getKP = (kps, name) => {
  const idx = POINTS[name];
  return (kps && idx !== undefined) ? kps[idx] : null;
};
const confident = (kp) => kp && (kp.score === undefined || kp.score > 0.35);

// ---------- Pose-specific rule engine (returns array of messages) ----------
function getCorrectionsForPose(poseName, keypoints, lang) {
  const s = STRINGS[lang];
  const msgs = [];

  // convenience kps
  const L_SH = getKP(keypoints, 'LEFT_SHOULDER');
  const R_SH = getKP(keypoints, 'RIGHT_SHOULDER');
  const L_EL = getKP(keypoints, 'LEFT_ELBOW');
  const R_EL = getKP(keypoints, 'RIGHT_ELBOW');
  const L_WR = getKP(keypoints, 'LEFT_WRIST');
  const R_WR = getKP(keypoints, 'RIGHT_WRIST');
  const L_HI = getKP(keypoints, 'LEFT_HIP');
  const R_HI = getKP(keypoints, 'RIGHT_HIP');
  const L_KN = getKP(keypoints, 'LEFT_KNEE');
  const R_KN = getKP(keypoints, 'RIGHT_KNEE');
  const L_AN = getKP(keypoints, 'LEFT_ANKLE');
  const R_AN = getKP(keypoints, 'RIGHT_ANKLE');

  // early bail if core points missing
  if (!confident(L_SH) || !confident(R_SH) || !confident(L_HI) || !confident(R_HI)) {
    return msgs;
  }

  const leftKneeAngle = confident(L_HI) && confident(L_KN) && confident(L_AN) ? angleBetween(L_HI, L_KN, L_AN) : null;
  const rightKneeAngle = confident(R_HI) && confident(R_KN) && confident(R_AN) ? angleBetween(R_HI, R_KN, R_AN) : null;
  const leftArmAngle = confident(L_SH) && confident(L_EL) && confident(L_WR) ? angleBetween(L_SH, L_EL, L_WR) : null;
  const rightArmAngle = confident(R_SH) && confident(R_EL) && confident(R_WR) ? angleBetween(R_SH, R_EL, R_WR) : null;

  const chestY = (L_SH.y + R_SH.y) / 2;
  const hipsY = (L_HI.y + R_HI.y) / 2;

  // helpers
  const wristAboveShoulder = (wr, sh) => confident(wr) && confident(sh) && wr.y < sh.y - 20;
  const wristNearAnkle = (wr, an) => confident(wr) && confident(an) && Math.abs(wr.y - an.y) < 40;

  switch (poseName) {
    case 'Tree': {
      // Expect one knee bent (angle < 140) and standing leg nearly straight (>165)
      const leftBent = leftKneeAngle !== null && leftKneeAngle < 140;
      const rightBent = rightKneeAngle !== null && rightKneeAngle < 140;
      if (!leftBent && !rightBent) {
        msgs.push(s.liftLegToThigh);
      } else {
        // standing leg must be straight
        if (leftBent && rightKneeAngle !== null && rightKneeAngle <= 165) msgs.push(s.straightenRightKnee || STRINGS[lang].keepLegsStraight);
        if (rightBent && leftKneeAngle !== null && leftKneeAngle <= 165) msgs.push(s.straightenLeftKnee || STRINGS[lang].keepLegsStraight);
      }
      // hands should not hang low
      if (confident(L_WR) && L_WR.y > chestY + 20) msgs.push(s.raiseLeft);
      if (confident(R_WR) && R_WR.y > chestY + 20) msgs.push(s.raiseRight);
      break;
    }

    case 'Chair': {
      // both knees bent ~ < 110 desired; if > 120 ask to bend more
      if (leftKneeAngle !== null && leftKneeAngle > 120) msgs.push(s.bendLeftKnee);
      if (rightKneeAngle !== null && rightKneeAngle > 120) msgs.push(s.bendRightKnee);
      // arms ideally up and straight
      if (!(leftArmAngle !== null && leftArmAngle > 160 && wristAboveShoulder(L_WR, L_SH))) msgs.push(s.raiseLeft);
      if (!(rightArmAngle !== null && rightArmAngle > 160 && wristAboveShoulder(R_WR, R_SH))) msgs.push(s.raiseRight);
      break;
    }

    case 'Cobra': {
      // chest should be lifted: shoulders above hips
      if (!(hipsY && chestY && chestY < hipsY - 10)) msgs.push(s.liftChest);
      // arms should be fairly straight
      if (leftArmAngle !== null && leftArmAngle < 150) msgs.push(s.straightenLeftArm || s.straightenArm);
      if (rightArmAngle !== null && rightArmAngle < 150) msgs.push(s.straightenRightArm || s.straightenArm);
      break;
    }

    case 'Warrior': {
      // one front knee bent (angle smaller than the other)
      if (leftKneeAngle !== null && rightKneeAngle !== null) {
        const leftIsFront = leftKneeAngle < rightKneeAngle;
        if (leftIsFront) {
          if (leftKneeAngle > 110) msgs.push(s.bendLeftKnee);
          if (rightKneeAngle < 165) msgs.push(s.straightenRightKnee);
        } else {
          if (rightKneeAngle > 110) msgs.push(s.bendRightKnee);
          if (leftKneeAngle < 165) msgs.push(s.straightenLeftKnee);
        }
      }
      // arms horizontal (y near shoulder y)
      if (!(confident(L_WR) && Math.abs(L_WR.y - L_SH.y) < 30 && confident(R_WR) && Math.abs(R_WR.y - R_SH.y) < 30)) {
        msgs.push(s.armsHorizontal);
      }
      break;
    }

    case 'Dog': {
      // hips should be high (smaller y than shoulders)
      if (!(hipsY && chestY && hipsY < chestY - 20)) msgs.push(s.liftHips);
      // arms and legs straight
      if (leftArmAngle !== null && leftArmAngle < 160) msgs.push(s.straightenLeftArm || s.straightenArm);
      if (rightArmAngle !== null && rightArmAngle < 160) msgs.push(s.straightenRightArm || s.straightenArm);
      if (leftKneeAngle !== null && leftKneeAngle < 165) msgs.push(s.keepLegsStraight);
      if (rightKneeAngle !== null && rightKneeAngle < 165) msgs.push(s.keepLegsStraight);
      break;
    }

    case 'Shoulderstand': {
      // legs vertical (knees straight), hips roughly over shoulders
      if (leftKneeAngle !== null && leftKneeAngle < 170) msgs.push(s.straightenLeftKnee);
      if (rightKneeAngle !== null && rightKneeAngle < 170) msgs.push(s.straightenRightKnee);
      if (!(Math.abs(((L_HI.y + R_HI.y) / 2) - ((L_SH.y + R_SH.y) / 2)) < 80)) msgs.push(s.alignHipsShoulders || s.stackHipsShoulders);
      break;
    }

    case 'Traingle': {
      // legs straight
      if (leftKneeAngle !== null && leftKneeAngle < 170) msgs.push(s.straightenLeftKnee);
      if (rightKneeAngle !== null && rightKneeAngle < 170) msgs.push(s.straightenRightKnee);
      // one arm up check (wrist above shoulder)
      const upArmOk = (confident(L_WR) && L_WR.y < L_SH.y - 20) || (confident(R_WR) && R_WR.y < R_SH.y - 20);
      if (!upArmOk) msgs.push(s.reachTopArm);
      break;
    }

    default:
      break;
  }

  return msgs;
}

// ---------- Main component ----------
export default function Yoga() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const [startingTime, setStartingTime] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [poseTime, setPoseTime] = useState(0);
  const [bestPerform, setBestPerform] = useState(0);
  const [currentPose, setCurrentPose] = useState('Tree');
  const [isStartPose, setIsStartPose] = useState(false);
  const [language, setLanguage] = useState('en');

  useEffect(() => {
    const timeDiff = (currentTime - startingTime) / 1000;
    if (flag) setPoseTime(timeDiff);
    if (timeDiff > bestPerform) setBestPerform(timeDiff);
  }, [currentTime]);

  useEffect(() => {
    setCurrentTime(0); setPoseTime(0); setBestPerform(0);
    lastSpokenText = '';
    lastSpokenAt = 0;
  }, [currentPose, language]);

  const CLASS_NO = {
    Chair: 0,
    Cobra: 1,
    Dog: 2,
    No_Pose: 3,
    Shoulderstand: 4,
    Traingle: 5,
    Tree: 6,
    Warrior: 7,
  };

  // run movenet & load classifier
  async function runMovenet() {
    const detectorConfig = { modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER };
    const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
    // load your classifier model (keep using your hosted model)
    const poseClassifier = await tf.loadLayersModel('https://models.s3.jp-tok.cloud-object-storage.appdomain.cloud/model.json');
    const countAudio = new Audio(count);
    countAudio.loop = true;

    interval = setInterval(() => {
      detectPose(detector, poseClassifier, countAudio);
    }, 100);
  }

  async function detectPose(detector, poseClassifier, countAudio) {
    if (!webcamRef.current || !webcamRef.current.video) return;
    if (webcamRef.current.video.readyState !== 4) return;

    const video = webcamRef.current.video;
    const pose = await detector.estimatePoses(video);
    if (!pose || !pose[0]) return;
    const keypoints = pose[0].keypoints;

    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    try {
      let notDetected = 0;
      keypoints.forEach((kp) => {
        if (kp.score > 0.35) {
          if (!(kp.name === 'left_eye' || kp.name === 'right_eye')) {
            drawPoint(ctx, kp.x, kp.y, 8, 'rgb(255,255,255)');
            const conns = keypointConnections[kp.name] || [];
            try {
              conns.forEach((c) => {
                const other = keypoints[POINTS[c.toUpperCase()]];
                if (other && other.x && other.y) {
                  drawSegment(ctx, [kp.x, kp.y], [other.x, other.y], skeletonColor);
                }
              });
            } catch (e) { /* ignore draw errors */ }
          }
        } else notDetected++;
      });

      if (notDetected > 6) { // not enough keypoints
        skeletonColor = 'rgb(255,255,255)';
        return;
      }

      // run classifier on normalized embedding
      const inputArr = keypoints.map(k => [k.x, k.y]); // 17 x 2
      const processedInput = landmarks_to_embedding(inputArr); // tf tensor [1,34]
      let classification = null;
      try {
        classification = poseClassifier.predict(processedInput);
      } catch (err) {
        // classifier might occasionally error; ignore and fallback to rules
        classification = null;
      }

      const feedbackMsgs = getCorrectionsForPose(currentPose, keypoints, language);

      // if classifier exists, use threshold; else rely purely on rules
      const threshold = 0.90;
      let classifiedStrong = false;
      if (classification) {
        try {
          const arr = await classification.array();
          const classNo = CLASS_NO[currentPose];
          if (arr && arr[0] && typeof arr[0][classNo] === 'number' && arr[0][classNo] > threshold) {
            classifiedStrong = true;
          }
        } catch (e) {
          // ignore
        }
      }

      if (classifiedStrong) {
        // positive detection: if no corrections -> positive voice; else speak first correction
        skeletonColor = 'rgb(0,255,0)';
        if (!flag) {
          countAudio.play();
          setStartingTime(new Date().getTime());
          flag = true;
        }
        setCurrentTime(new Date().getTime());
        if (feedbackMsgs.length === 0) {
          speakText(STRINGS[language].correct, language);
        } else {
          speakText(feedbackMsgs[0], language);
        }
      } else {
        // not strongly classified: still give helpful guidance if rules suggest something
        skeletonColor = 'rgb(255,255,255)';
        flag = false;
        try { countAudio.pause(); countAudio.currentTime = 0; } catch (e) {}
        if (feedbackMsgs.length > 0) {
          speakText(feedbackMsgs[0], language);
        }
      }

      // cleanup tf tensors
      try { processedInput.dispose(); if (classification && classification.dispose) classification.dispose(); } catch(e) {}
    } catch (err) {
      console.error(err);
    }
  }

  function startYoga() {
    // trigger voice list loading in some browsers
    window.speechSynthesis.getVoices();
    setIsStartPose(true);
    speakText(language === 'ta' ? "யோகா துவங்குகிறது" : "Starting yoga", language);
    runMovenet();
  }

  function stopPose() {
    setIsStartPose(false);
    clearInterval(interval);
    window.speechSynthesis.cancel();
  }

  // UI
  if (isStartPose) {
    return (
      <div className="yoga-container">
        <div className="performance-container">
          <div className="pose-performance"><h4>Pose Time: {poseTime} s</h4></div>
          <div className="pose-performance"><h4>Best: {bestPerform} s</h4></div>
        </div>

        <div style={{ position: 'absolute', top: 60, left: 120, zIndex: 3 }}>
          <label style={{ marginRight: 8 }}>Language:</label>
          <select value={language} onChange={e => setLanguage(e.target.value)}>
            <option value="en">English</option>
            <option value="ta">தமிழ்</option>
          </select>
        </div>

        <div>
          <Webcam
            width='640px'
            height='480px'
            id="webcam"
            ref={webcamRef}
            style={{ position: 'absolute', left: 120, top: 100, padding: '0px' }}
          />
          <canvas ref={canvasRef} id="my-canvas" width='640px' height='480px'
            style={{ position: 'absolute', left: 120, top: 100, zIndex: 1 }} />
          <div><img src={poseImages[currentPose]} className="pose-img" alt="pose" /></div>
        </div>

        <button onClick={stopPose} className="secondary-btn">Stop Pose</button>
      </div>
    );
  }

  return (
    <div className="yoga-container">
      <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
        <DropDown poseList={poseList} currentPose={currentPose} setCurrentPose={setCurrentPose} />
        <div>
          <label style={{ marginRight: 8 }}>Language:</label>
          <select value={language} onChange={e => setLanguage(e.target.value)}>
            <option value="en">English</option>
            <option value="ta">தமிழ்</option>
          </select>
        </div>
      </div>

      <Instructions currentPose={currentPose} />
      <button onClick={startYoga} className="secondary-btn">Start Pose</button>
    </div>
  );
}
