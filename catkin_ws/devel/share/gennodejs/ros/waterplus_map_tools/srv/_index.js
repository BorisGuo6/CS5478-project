
"use strict";

let GetNumOfWaypoints = require('./GetNumOfWaypoints.js')
let AddNewWaypoint = require('./AddNewWaypoint.js')
let GetWaypointByName = require('./GetWaypointByName.js')
let GetWaypointByIndex = require('./GetWaypointByIndex.js')
let SaveWaypoints = require('./SaveWaypoints.js')
let GetChargerByName = require('./GetChargerByName.js')

module.exports = {
  GetNumOfWaypoints: GetNumOfWaypoints,
  AddNewWaypoint: AddNewWaypoint,
  GetWaypointByName: GetWaypointByName,
  GetWaypointByIndex: GetWaypointByIndex,
  SaveWaypoints: SaveWaypoints,
  GetChargerByName: GetChargerByName,
};
