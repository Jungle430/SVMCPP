--
-- Table structure for table `svm_param`
--

DROP TABLE IF EXISTS `svm_param`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `svm_param` (
  `number` int(11) NOT NULL COMMENT '模型所对数字',
  `alpha` text NOT NULL COMMENT 'alpha向量',
  `b` double NOT NULL COMMENT 'b参数',
  `update_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最后更新时间',
  PRIMARY KEY (`number`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='svm参数';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `svm_param`
--

LOCK TABLES `svm_param` WRITE;
/*!40000 ALTER TABLE `svm_param` DISABLE KEYS */;
/*!40000 ALTER TABLE `svm_param` ENABLE KEYS */;
UNLOCK TABLES;
