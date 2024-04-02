/*
 Navicat Premium Data Transfer

 Source Server         : demo
 Source Server Type    : MySQL
 Source Server Version : 50741
 Source Host           : localhost:3306
 Source Schema         : demo

 Target Server Type    : MySQL
 Target Server Version : 50741
 File Encoding         : 65001

 Date: 02/04/2024 16:10:48
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for hardwaresetting
-- ----------------------------
DROP TABLE IF EXISTS `hardwaresetting`;
CREATE TABLE `hardwaresetting`  (
  `HardwareSettingId` int(11) NOT NULL AUTO_INCREMENT,
  `HardwareName` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`HardwareSettingId`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 3 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of hardwaresetting
-- ----------------------------
INSERT INTO `hardwaresetting` VALUES (1, '摄像头,烟雾传感器,驱动系统');
INSERT INTO `hardwaresetting` VALUES (2, '仅摄像头');

-- ----------------------------
-- Table structure for importantpic
-- ----------------------------
DROP TABLE IF EXISTS `importantpic`;
CREATE TABLE `importantpic`  (
  `ImportantPicId` int(11) NOT NULL AUTO_INCREMENT,
  `UserId` int(11) NULL DEFAULT NULL,
  `Message` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `isRead` varchar(1) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `setTime` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`ImportantPicId`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of importantpic
-- ----------------------------

-- ----------------------------
-- Table structure for screenshotrecord
-- ----------------------------
DROP TABLE IF EXISTS `screenshotrecord`;
CREATE TABLE `screenshotrecord`  (
  `ScreenshotRecordId` int(11) NOT NULL AUTO_INCREMENT,
  `UserId` int(11) NULL DEFAULT NULL,
  `TerritoryId` int(11) NULL DEFAULT NULL,
  `ScreenshotName` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `ScreenshotPath` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `IsImportant` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`ScreenshotRecordId`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of screenshotrecord
-- ----------------------------

-- ----------------------------
-- Table structure for smokingrecord
-- ----------------------------
DROP TABLE IF EXISTS `smokingrecord`;
CREATE TABLE `smokingrecord`  (
  `RecordId` int(11) NOT NULL AUTO_INCREMENT,
  `TerritoryId` int(11) NOT NULL,
  `SmokeStartTime` timestamp NULL DEFAULT NULL,
  `ConfidenceLevel` double(5, 2) NULL DEFAULT NULL,
  `ScreenshotRecordId` int(11) NULL DEFAULT NULL,
  PRIMARY KEY (`RecordId`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of smokingrecord
-- ----------------------------

-- ----------------------------
-- Table structure for territory
-- ----------------------------
DROP TABLE IF EXISTS `territory`;
CREATE TABLE `territory`  (
  `TerritoryId` int(11) NOT NULL AUTO_INCREMENT,
  `TerritoryName` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `HardwareSettingId` int(11) NULL DEFAULT NULL,
  `TerritoryConfigurationId` int(11) NULL DEFAULT NULL,
  `StorageSize` double NULL DEFAULT NULL,
  `ConfidenceLevel` double NULL DEFAULT NULL,
  PRIMARY KEY (`TerritoryId`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 14 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of territory
-- ----------------------------
INSERT INTO `territory` VALUES (8, '主房间', 2, 1, 100, 0.6);
INSERT INTO `territory` VALUES (9, '附卧室', 2, 1, 100, 0.6);
INSERT INTO `territory` VALUES (10, '二期', 1, 1, 50, 0.6);
INSERT INTO `territory` VALUES (11, '三期', 1, 1, 30, 0.6);
INSERT INTO `territory` VALUES (12, '宿舍', 1, 1, 30, 0.6);
INSERT INTO `territory` VALUES (13, '食堂', 1, 1, 30, 0.6);

-- ----------------------------
-- Table structure for territorychangerequest
-- ----------------------------
DROP TABLE IF EXISTS `territorychangerequest`;
CREATE TABLE `territorychangerequest`  (
  `ChangeRequestId` int(11) NOT NULL AUTO_INCREMENT,
  `UserId` int(11) NOT NULL COMMENT '用户id',
  `RequestedTerritoryId` int(11) NOT NULL COMMENT '需求的id',
  `RequestDate` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '请求开始时间',
  `ApprovalDate` timestamp NULL DEFAULT NULL COMMENT '批准日期',
  `ApproverId` int(11) NULL DEFAULT NULL COMMENT '批准者id',
  `Remarks` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '备注',
  `RequestStatus` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '状态',
  `territoryConfigurationId` int(11) NULL DEFAULT NULL,
  PRIMARY KEY (`ChangeRequestId`) USING BTREE,
  INDEX `idx_userid`(`UserId`) USING BTREE,
  INDEX `idx_requestedterritoryid`(`RequestedTerritoryId`) USING BTREE,
  INDEX `idx_approverid`(`ApproverId`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 8 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of territorychangerequest
-- ----------------------------
INSERT INTO `territorychangerequest` VALUES (1, 2, 8, '2024-03-13 12:01:04', '2024-03-14 00:00:00', 1, NULL, 'refuse', 1);
INSERT INTO `territorychangerequest` VALUES (2, 2, 10, '2024-03-15 15:48:19', '2024-03-15 00:00:00', 1, NULL, 'agree', 2);
INSERT INTO `territorychangerequest` VALUES (3, 2, 9, '2024-03-15 15:58:03', '2024-03-15 00:00:00', 1, NULL, 'agree', 2);
INSERT INTO `territorychangerequest` VALUES (4, 2, 11, '2024-03-17 14:43:23', '2024-03-17 00:00:00', 1, NULL, 'agree', 1);
INSERT INTO `territorychangerequest` VALUES (5, 2, 10, '2024-03-17 15:36:24', '2024-03-17 00:00:00', 1, NULL, 'refuse', 2);
INSERT INTO `territorychangerequest` VALUES (6, 2, 10, '2024-03-17 15:39:16', '2024-03-17 00:00:00', 1, NULL, 'agree', 2);
INSERT INTO `territorychangerequest` VALUES (7, 2, 12, '2024-03-17 15:42:55', '2024-03-17 00:00:00', 1, NULL, 'refuse', 2);

-- ----------------------------
-- Table structure for territoryconfiguration
-- ----------------------------
DROP TABLE IF EXISTS `territoryconfiguration`;
CREATE TABLE `territoryconfiguration`  (
  `TerritoryConfigurationId` int(11) NOT NULL AUTO_INCREMENT,
  `Action` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`TerritoryConfigurationId`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 3 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of territoryconfiguration
-- ----------------------------
INSERT INTO `territoryconfiguration` VALUES (1, '静默记录');
INSERT INTO `territoryconfiguration` VALUES (2, '仅通知管理员');

-- ----------------------------
-- Table structure for user
-- ----------------------------
DROP TABLE IF EXISTS `user`;
CREATE TABLE `user`  (
  `UserID` int(11) NOT NULL AUTO_INCREMENT,
  `Username` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `Role` varchar(1) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `Password` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `Salt` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `Telephone` varchar(14) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `regTime` datetime NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
  `enabled` varchar(1) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`UserID`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 5 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of user
-- ----------------------------
INSERT INTO `user` VALUES (1, 'wzy', '1', '1', '1', '114', NULL, '1');
INSERT INTO `user` VALUES (2, 'wzy1', '0', '1', '123', '114', '2024-02-14 21:27:36', '1');
INSERT INTO `user` VALUES (3, 'wzy1213', '0', '12', '123', '114', '2024-02-14 21:31:42', '1');
INSERT INTO `user` VALUES (4, 'wzy11', '0', '1', '123', '13052765120', NULL, '1');

-- ----------------------------
-- Table structure for userfavorite
-- ----------------------------
DROP TABLE IF EXISTS `userfavorite`;
CREATE TABLE `userfavorite`  (
  `FavoriteId` int(11) NOT NULL AUTO_INCREMENT,
  `UserId` int(11) NULL DEFAULT NULL,
  `ScreenshotRecordId` int(11) NULL DEFAULT NULL,
  PRIMARY KEY (`FavoriteId`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of userfavorite
-- ----------------------------

-- ----------------------------
-- Table structure for userterritory
-- ----------------------------
DROP TABLE IF EXISTS `userterritory`;
CREATE TABLE `userterritory`  (
  `Id` int(11) NOT NULL AUTO_INCREMENT,
  `UserId` int(11) NULL DEFAULT NULL,
  `TerritoryId` int(11) NULL DEFAULT NULL,
  PRIMARY KEY (`Id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 12 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of userterritory
-- ----------------------------
INSERT INTO `userterritory` VALUES (10, 2, 9);
INSERT INTO `userterritory` VALUES (11, 2, 11);

SET FOREIGN_KEY_CHECKS = 1;
