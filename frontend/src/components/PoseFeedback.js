// src/components/PoseFeedback.js
import React from "react";

function getTrianglePoseFeedback(pose) {
  if (!pose.leftHandUp) {
    return "Raise your left hand up";
  } else if (!pose.rightHandDown) {
    return "Lower your right hand down";
  } else if (!pose.feetApart) {
    return "Widen your stance";
  }
  return "Everything was correct, continue like that!";
}

function getCobraPoseFeedback(pose) {
  if (!pose.chestLifted) {
    return "Lift your chest slightly more";
  } else if (!pose.elbowsClose) {
    return "Keep your elbows closer to the body";
  }
  return "Everything was correct, continue like that!";
}

function getTreePoseFeedback(pose) {
  if (!pose.balanceStable) {
    return "Try to balance more steadily";
  } else if (!pose.handsTogether) {
    return "Bring your palms together above your head";
  }
  return "Everything was correct, continue like that!";
}

// 🔑 Main feedback handler
export default function PoseFeedback({ poseName, poseData }) {
  let feedback = "";

  switch (poseName.toLowerCase()) {
    case "triangle":
      feedback = getTrianglePoseFeedback(poseData);
      break;
    case "cobra":
      feedback = getCobraPoseFeedback(poseData);
      break;
    case "tree":
      feedback = getTreePoseFeedback(poseData);
      break;
    default:
      feedback = "Pose not recognized.";
  }

  return (
    <div className="mt-4 p-4 bg-white shadow-md rounded-lg text-lg font-semibold text-gray-800">
      {feedback}
    </div>
  );
}
